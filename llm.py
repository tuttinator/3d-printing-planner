import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Literal

from state import Provider


ThinkingLevel = Literal["LOW", "MEDIUM", "HIGH"]


@dataclass(slots=True)
class FunctionCall:
    name: str
    args: dict[str, Any]
    call_id: str | None = None


@dataclass(slots=True)
class FunctionResponse:
    name: str
    response: dict[str, Any]
    call_id: str | None = None


@dataclass(slots=True)
class MessagePart:
    text: str | None = None
    function_call: FunctionCall | None = None
    function_response: FunctionResponse | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "function_call": (
                asdict(self.function_call) if self.function_call is not None else None
            ),
            "function_response": (
                asdict(self.function_response)
                if self.function_response is not None
                else None
            ),
        }


@dataclass(slots=True)
class Message:
    role: Literal["assistant", "tool", "user"]
    parts: list[MessagePart]

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "parts": [part.to_dict() for part in self.parts]}


@dataclass(slots=True)
class ModelGenerateConfig:
    system_instruction: str
    tools: list[dict[str, Any]]
    thinking_level: ThinkingLevel

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelClient(ABC):
    @abstractmethod
    async def generate_content(
        self,
        *,
        model: str,
        contents: list[Message],
        config: ModelGenerateConfig,
    ) -> Message:
        raise NotImplementedError


def get_default_model(provider: Provider) -> str:
    return {
        "anthropic": "claude-sonnet-4-20250514",
        "gemini": "gemini-2.5-pro",
        "openai": "gpt-5",
    }[provider]


def get_api_key(provider: Provider) -> str:
    env_var = {
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
    }[provider]
    value = os.getenv(env_var)
    if not value:
        raise RuntimeError(f"{env_var} is required for provider '{provider}'.")
    return value


def build_model_client(provider: Provider) -> ModelClient:
    if provider == "openai":
        return OpenAIModelClient(api_key=get_api_key(provider))
    if provider == "anthropic":
        return AnthropicModelClient(api_key=get_api_key(provider))
    return GeminiModelClient(api_key=get_api_key(provider))


class OpenAIModelClient(ModelClient):
    def __init__(self, *, api_key: str) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for provider 'openai'."
            ) from exc

        self.client = AsyncOpenAI(api_key=api_key)

    async def generate_content(
        self,
        *,
        model: str,
        contents: list[Message],
        config: ModelGenerateConfig,
    ) -> Message:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": config.system_instruction}
        ]
        for content in contents:
            messages.extend(_to_openai_messages(content))

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=_to_openai_tools(config.tools) or None,
        )
        message = response.choices[0].message

        parts: list[MessagePart] = []
        if message.content:
            parts.append(MessagePart(text=message.content))
        for tool_call in message.tool_calls or []:
            arguments = tool_call.function.arguments or "{}"
            parts.append(
                MessagePart(
                    function_call=FunctionCall(
                        name=tool_call.function.name,
                        args=json.loads(arguments),
                        call_id=tool_call.id,
                    )
                )
            )

        return Message(role="assistant", parts=parts)


class AnthropicModelClient(ModelClient):
    def __init__(self, *, api_key: str) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "The 'anthropic' package is required for provider 'anthropic'."
            ) from exc

        self.client = AsyncAnthropic(api_key=api_key)

    async def generate_content(
        self,
        *,
        model: str,
        contents: list[Message],
        config: ModelGenerateConfig,
    ) -> Message:
        messages = [_to_anthropic_message(content) for content in contents]
        response = await self.client.messages.create(
            model=model,
            max_tokens=4096,
            system=config.system_instruction,
            messages=messages,
            tools=_to_anthropic_tools(config.tools) or [],
        )

        parts: list[MessagePart] = []
        for block in response.content:
            if block.type == "text":
                parts.append(MessagePart(text=block.text))
            elif block.type == "tool_use":
                parts.append(
                    MessagePart(
                        function_call=FunctionCall(
                            name=block.name,
                            args=dict(block.input),
                            call_id=block.id,
                        )
                    )
                )

        return Message(role="assistant", parts=parts)


class GeminiModelClient(ModelClient):
    def __init__(self, *, api_key: str) -> None:
        try:
            from google.genai import Client
        except ImportError as exc:
            raise RuntimeError(
                "The 'google-genai' package is required for provider 'gemini'."
            ) from exc

        self.client = Client(api_key=api_key)

    async def generate_content(
        self,
        *,
        model: str,
        contents: list[Message],
        config: ModelGenerateConfig,
    ) -> Message:
        from google.genai import types

        completion = await self.client.aio.models.generate_content(
            model=model,
            contents=[_to_gemini_content(content) for content in contents],
            config=types.GenerateContentConfig(
                tools=[_to_gemini_tool(tool) for tool in config.tools],
                system_instruction=config.system_instruction,
                thinking_config=types.ThinkingConfig(
                    thinking_level=config.thinking_level
                ),
            ),
        )
        candidate = completion.candidates[0].content
        parts: list[MessagePart] = []
        for part in candidate.parts:
            if part.text:
                parts.append(MessagePart(text=part.text))
            elif part.function_call:
                parts.append(
                    MessagePart(
                        function_call=FunctionCall(
                            name=part.function_call.name,
                            args=dict(part.function_call.args or {}),
                            call_id=getattr(part.function_call, "id", None),
                        )
                    )
                )
        return Message(role="assistant", parts=parts)


def _to_openai_messages(content: Message) -> list[dict[str, Any]]:
    if content.role == "tool":
        messages: list[dict[str, Any]] = []
        for part in content.parts:
            if part.function_response is None:
                continue
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": part.function_response.call_id,
                    "content": json.dumps(part.function_response.response),
                }
            )
        return messages

    text_parts = [part.text for part in content.parts if part.text]
    function_calls = [
        part.function_call for part in content.parts if part.function_call is not None
    ]
    message: dict[str, Any] = {
        "role": "assistant" if content.role == "assistant" else "user",
    }
    if text_parts:
        message["content"] = "\n".join(text_parts)
    elif content.role != "assistant":
        message["content"] = ""

    if function_calls:
        message["tool_calls"] = [
            {
                "id": call.call_id or call.name,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.args),
                },
            }
            for call in function_calls
        ]
    return [message]


def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in tools
    ]


def _to_anthropic_message(content: Message) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for part in content.parts:
        if part.text:
            blocks.append({"type": "text", "text": part.text})
        if part.function_call is not None:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": part.function_call.call_id or part.function_call.name,
                    "name": part.function_call.name,
                    "input": part.function_call.args,
                }
            )
        if part.function_response is not None:
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": (
                        part.function_response.call_id or part.function_response.name
                    ),
                    "content": json.dumps(part.function_response.response),
                }
            )

    role = "assistant" if content.role == "assistant" else "user"
    return {"role": role, "content": blocks}


def _to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"],
        }
        for tool in tools
    ]


def _to_gemini_content(message: Message) -> Any:
    from google.genai import types

    role = "model" if message.role == "assistant" else "user"
    parts: list[Any] = []
    for part in message.parts:
        if part.text:
            parts.append(types.Part.from_text(text=part.text))
        if part.function_call is not None:
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        name=part.function_call.name,
                        args=part.function_call.args,
                    )
                )
            )
        if part.function_response is not None:
            parts.append(
                types.Part.from_function_response(
                    name=part.function_response.name,
                    response=part.function_response.response,
                )
            )
    return types.Content(role=role, parts=parts)


def _to_gemini_tool(tool: dict[str, Any]) -> Any:
    from google.genai import types

    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=_schema_to_gemini_schema(tool["parameters"]),
            )
        ]
    )


def _schema_to_gemini_schema(schema: dict[str, Any]) -> Any:
    from google.genai import types

    properties = schema.get("properties", {})
    converted_properties = {
        name: _schema_to_gemini_schema(value) for name, value in properties.items()
    }
    items = schema.get("items")

    return types.Schema(
        type=_normalize_gemini_type(schema.get("type")),
        description=schema.get("description"),
        enum=schema.get("enum"),
        nullable=schema.get("nullable"),
        properties=converted_properties or None,
        required=schema.get("required"),
        items=_schema_to_gemini_schema(items) if isinstance(items, dict) else None,
    )


def _normalize_gemini_type(value: Any) -> Any:
    if isinstance(value, str):
        return value.upper()
    return value
