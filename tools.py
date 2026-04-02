import asyncio
import base64
import os
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, Field, field_validator

from state import AgentContext, RunState

ArgsT = TypeVar("ArgsT", bound=BaseModel)
MAX_DELEGATED_QUERIES = 3
DOCKER_TIMEOUT_SECONDS = 120


@dataclass(slots=True)
class ReadFileMetadata:
    path: str
    contents: str


@dataclass(slots=True)
class WriteFileMetadata:
    path: str
    contents: str


@dataclass(slots=True)
class EditFileMetadata:
    path: str
    old_text: str
    new_text: str


@dataclass(slots=True)
class ModifyTodoMetadata:
    action: Literal["add", "remove"]
    todos: list[str]


@dataclass(slots=True)
class SearchWebMetadata:
    query: str
    raw_results: Any


@dataclass(slots=True)
class DelegateSearchMetadata:
    queries: list[str]
    results: list[dict[str, str]]


@dataclass(slots=True)
class GeneratePlanMetadata:
    todos: list[str]


@dataclass(slots=True)
class BashMetadata:
    command: str
    returncode: int
    stdout: str
    stderr: str


@dataclass(slots=True)
class OpenScadMetadata:
    action: Literal["export_stl", "render_scad", "validate_scad"]
    input_path: str
    output_path: str | None
    command: str
    returncode: int
    stdout: str
    stderr: str


@dataclass(slots=True)
class ConceptImageMetadata:
    provider: Literal["gemini", "openai"]
    model: str
    output_path: str
    prompt: str


ToolMetadata: TypeAlias = (
    BashMetadata
    | DelegateSearchMetadata
    | EditFileMetadata
    | GeneratePlanMetadata
    | ModifyTodoMetadata
    | OpenScadMetadata
    | ConceptImageMetadata
    | ReadFileMetadata
    | SearchWebMetadata
    | WriteFileMetadata
)


@dataclass(slots=True)
class ToolExecutionResult:
    model_response: dict[str, Any]
    metadata: ToolMetadata | None = None


ToolHandler = Callable[[ArgsT, RunState, AgentContext], Awaitable[ToolExecutionResult]]


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler

    def to_tool_schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": schema["properties"],
                "required": schema.get("required", []),
            },
        }


class ReadFileArgs(BaseModel):
    path: str


class WriteFileArgs(BaseModel):
    path: str
    contents: str


class EditFileArgs(BaseModel):
    path: str
    old_text: str = Field(
        ...,
        min_length=1,
        description="The exact text to replace.",
    )
    new_text: str = Field(
        ...,
        description="The replacement text.",
    )


def _resolve_workspace_path(path: str, context: AgentContext) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = context.workspace_root / candidate
    resolved = candidate.resolve()
    workspace = context.workspace_root.resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"Path must stay within the workspace: {path}")
    return resolved


def _workspace_relative_path(path: Path, context: AgentContext) -> str:
    return path.resolve().relative_to(context.workspace_root.resolve()).as_posix()


async def read_file(
    args: ReadFileArgs,
    state: RunState,
    context: AgentContext,
) -> ToolExecutionResult:
    del state

    try:
        path = _resolve_workspace_path(args.path, context)
    except ValueError as error:
        return ToolExecutionResult(model_response={"error": str(error)})

    if not path.exists() or not path.is_file():
        return ToolExecutionResult(
            model_response={"error": f"File does not exist: {args.path}"}
        )

    contents = path.read_text(encoding="utf-8")
    relative_path = _workspace_relative_path(path, context)
    return ToolExecutionResult(
        model_response={
            "result": f"Read file at path {relative_path}",
            "path": relative_path,
            "contents": contents,
        },
        metadata=ReadFileMetadata(path=relative_path, contents=contents),
    )


async def write_file(
    args: WriteFileArgs,
    state: RunState,
    context: AgentContext,
) -> ToolExecutionResult:
    del state

    try:
        path = _resolve_workspace_path(args.path, context)
    except ValueError as error:
        return ToolExecutionResult(model_response={"error": str(error)})

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args.contents, encoding="utf-8")
    relative_path = _workspace_relative_path(path, context)

    return ToolExecutionResult(
        model_response={
            "result": f"Wrote file at path {relative_path}",
            "path": relative_path,
        },
        metadata=WriteFileMetadata(path=relative_path, contents=args.contents),
    )


async def edit_file(
    args: EditFileArgs,
    state: RunState,
    context: AgentContext,
) -> ToolExecutionResult:
    del state

    try:
        path = _resolve_workspace_path(args.path, context)
    except ValueError as error:
        return ToolExecutionResult(model_response={"error": str(error)})

    if not path.exists() or not path.is_file():
        return ToolExecutionResult(
            model_response={"error": f"File does not exist: {args.path}"}
        )

    contents = path.read_text(encoding="utf-8")
    if args.old_text not in contents:
        return ToolExecutionResult(
            model_response={
                "error": (
                    f"Could not find the requested text to replace in {args.path}"
                )
            }
        )

    updated_contents = contents.replace(args.old_text, args.new_text, 1)
    path.write_text(updated_contents, encoding="utf-8")
    relative_path = _workspace_relative_path(path, context)

    return ToolExecutionResult(
        model_response={
            "result": f"Edited file at path {relative_path}",
            "path": relative_path,
        },
        metadata=EditFileMetadata(
            path=relative_path,
            old_text=args.old_text,
            new_text=args.new_text,
        ),
    )


class ModifyTodoArgs(BaseModel):
    action: Literal["add", "remove"]
    todos: list[str]


async def modify_todo(
    args: ModifyTodoArgs,
    state: RunState,
    context: AgentContext,
) -> ToolExecutionResult:
    del context

    if args.action == "add":
        state.add_todos(args.todos)
        return ToolExecutionResult(
            model_response={
                "result": "Todos updated.",
                "todos": list(state.todos),
            },
            metadata=ModifyTodoMetadata(action=args.action, todos=list(args.todos)),
        )

    requested = [todo.strip() for todo in args.todos]
    missing = []
    existing_lower = {todo.lower() for todo in state.todos}
    for todo in requested:
        if todo.lower() not in existing_lower:
            missing.append(todo)

    if missing:
        return ToolExecutionResult(
            model_response={"error": f"Todos not found: {', '.join(missing)}"}
        )

    state.remove_todos(args.todos)
    return ToolExecutionResult(
        model_response={"result": "Todos updated.", "todos": list(state.todos)},
        metadata=ModifyTodoMetadata(action=args.action, todos=list(args.todos)),
    )


class SearchWebArgs(BaseModel):
    query: str


async def search_web(
    args: SearchWebArgs,
    state: RunState,
    context: AgentContext,
) -> ToolExecutionResult:
    del state

    exa = context.exa
    if exa is None:
        return ToolExecutionResult(
            model_response={"error": "Exa client is not configured."}
        )

    results = exa.search(
        args.query,
        num_results=10,
        type="auto",
        contents={"highlights": {"max_characters": 4000}},
    )

    formatted_results: list[dict[str, Any]] = []
    for item in results.results:
        formatted_results.append(
            {
                "title": item.title or "",
                "url": item.url,
                "highlights": item.highlights or [],
            }
        )

    return ToolExecutionResult(
        model_response={
            "query": args.query,
            "results": formatted_results,
        },
        metadata=SearchWebMetadata(query=args.query, raw_results=results),
    )


class DelegateSearchArgs(BaseModel):
    queries: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_DELEGATED_QUERIES,
        description=(
            "A set of distinct search questions that you need answered. Each query "
            "should cover a meaningfully different angle."
        ),
    )

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, queries: list[str]) -> list[str]:
        normalized_queries: list[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized = " ".join(query.split())
            if not normalized:
                raise ValueError("Queries must not be empty.")
            normalized_key = normalized.lower()
            if normalized_key in seen:
                raise ValueError("Queries must be distinct.")
            seen.add(normalized_key)
            normalized_queries.append(normalized)
        return normalized_queries


async def delegate_search(
    args: DelegateSearchArgs,
    state: RunState,
    context: AgentContext,
) -> ToolExecutionResult:
    del state

    if context.search_agent_runner is None:
        return ToolExecutionResult(
            model_response={"error": "Search subagent runner is not configured."}
        )

    results = await context.search_agent_runner(args.queries)
    if not results:
        return ToolExecutionResult(
            model_response={"error": "Search subagent did not return any results."}
        )

    return ToolExecutionResult(
        model_response={
            "queries": list(args.queries),
            "results": results,
        },
        metadata=DelegateSearchMetadata(
            queries=list(args.queries),
            results=results,
        ),
    )


class BashArgs(BaseModel):
    command: str = Field(
        ...,
        min_length=1,
        description="A bash command to run in the current working directory.",
    )


async def bash(
    args: BashArgs,
    state: RunState,
    context: AgentContext,
) -> ToolExecutionResult:
    del state, context

    try:
        process = await asyncio.create_subprocess_shell(
            args.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=30,
        )
    except TimeoutError:
        return ToolExecutionResult(
            model_response={
                "error": f"Command timed out after 30 seconds: {args.command}"
            }
        )

    stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
    returncode = process.returncode or 0

    return ToolExecutionResult(
        model_response={
            "command": args.command,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
        },
        metadata=BashMetadata(
            command=args.command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        ),
    )


class ValidateScadArgs(BaseModel):
    path: str = Field(..., description="Path to the OpenSCAD file to validate.")


class ExportStlArgs(BaseModel):
    path: str = Field(..., description="Path to the OpenSCAD file to export.")
    output_path: str = Field(..., description="Destination STL path in the workspace.")


class RenderScadArgs(BaseModel):
    path: str = Field(..., description="Path to the OpenSCAD file to render.")
    output_path: str = Field(..., description="Destination PNG path in the workspace.")
    width: int = Field(1024, ge=64, le=4096)
    height: int = Field(768, ge=64, le=4096)
    view: Literal["axes", "crosshairs", "edges", "scales"] | None = None


class GenerateConceptImageArgs(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        description="Prompt describing the concept image to generate.",
    )
    output_path: str = Field(
        ...,
        description="Destination PNG path in the workspace.",
    )
    provider: Literal["auto", "gemini", "openai"] = Field(
        "auto",
        description="Image generation provider. Use auto to pick from the configured/default provider.",
    )


async def _run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int = 30,
    context: AgentContext | None = None,
    status_key: str | None = None,
    cleanup_command: list[str] | None = None,
) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    async def read_stream(
        stream: asyncio.StreamReader | None,
        chunks: list[str],
        label: str,
    ) -> None:
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace")
            chunks.append(text)
            stripped = text.strip()
            if stripped and context is not None and status_key is not None:
                _set_activity_status(
                    context, status_key, f"{label}: {truncate_status(stripped)}"
                )

    stdout_task = asyncio.create_task(
        read_stream(process.stdout, stdout_chunks, "stdout")
    )
    stderr_task = asyncio.create_task(
        read_stream(process.stderr, stderr_chunks, "stderr")
    )
    try:
        await asyncio.wait_for(
            process.wait(),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        if cleanup_command is not None:
            cleanup = await asyncio.create_subprocess_exec(
                *cleanup_command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await cleanup.communicate()
        await asyncio.gather(stdout_task, stderr_task)
        stdout = "".join(stdout_chunks).strip()
        stderr = "".join(stderr_chunks).strip()
        timeout_stderr = f"Command timed out after {timeout_seconds} seconds."
        if stderr:
            timeout_stderr = f"{timeout_stderr}\n{stderr}"
        return 124, stdout, timeout_stderr

    await asyncio.gather(stdout_task, stderr_task)
    stdout = "".join(stdout_chunks).strip()
    stderr = "".join(stderr_chunks).strip()
    return process.returncode or 0, stdout, stderr


def truncate_status(text: str, max_length: int = 72) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 3].rstrip()}..."


def _resolve_image_provider(
    requested: Literal["auto", "gemini", "openai"],
    context: AgentContext,
) -> Literal["gemini", "openai"]:
    if requested in {"gemini", "openai"}:
        return requested
    if context.preferred_image_provider in {"gemini", "openai"}:
        return context.preferred_image_provider
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    raise RuntimeError(
        "No image generation provider is configured. Set GEMINI_API_KEY or OPENAI_API_KEY."
    )


def _is_resource_exhausted_error(error: Exception) -> bool:
    text = str(error).lower()
    return (
        "429" in text
        or "resource_exhausted" in text
        or "resource exhausted" in text
        or "quota" in text
        or "rate limit" in text
    )


async def generate_concept_image(
    args: GenerateConceptImageArgs,
    state: RunState,
    context: AgentContext,
) -> ToolExecutionResult:
    del state
    if not args.output_path.lower().endswith(".png"):
        return ToolExecutionResult(
            model_response={"error": "Concept image output_path must end with .png."}
        )

    try:
        destination_path = _resolve_workspace_path(args.output_path, context)
    except ValueError as error:
        return ToolExecutionResult(model_response={"error": str(error)})

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    output_relative = _workspace_relative_path(destination_path, context)

    try:
        provider = _resolve_image_provider(args.provider, context)
    except RuntimeError as error:
        return ToolExecutionResult(model_response={"error": str(error)})

    status_key = f"image: {Path(output_relative).name}"
    _set_activity_status(context, status_key, f"Generating with {provider}")

    async def generate_with_openai() -> tuple[bytes, str, str]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.images.generate(
            model="gpt-image-1",
            prompt=args.prompt,
            size="1536x1024",
        )
        image_b64 = response.data[0].b64_json
        if not image_b64:
            raise RuntimeError("OpenAI image response did not contain image bytes.")
        return base64.b64decode(image_b64), "openai", "gpt-image-1"

    async def generate_with_gemini() -> tuple[bytes, str, str]:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        last_error: Exception | None = None
        for attempt, delay_seconds in enumerate((0, 5, 15), start=1):
            if delay_seconds:
                _set_activity_status(
                    context,
                    status_key,
                    f"Gemini retry {attempt}/3 after {delay_seconds}s backoff",
                )
                await asyncio.sleep(delay_seconds)
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model="gemini-2.5-flash-image",
                    contents=args.prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"]
                    ),
                )
                image_bytes: bytes | None = None
                for candidate in response.candidates or []:
                    content = getattr(candidate, "content", None)
                    if content is None:
                        continue
                    for part in content.parts or []:
                        inline_data = getattr(part, "inline_data", None)
                        if inline_data and getattr(inline_data, "data", None):
                            image_bytes = inline_data.data
                            break
                    if image_bytes is not None:
                        break
                if image_bytes is None:
                    raise RuntimeError(
                        "Gemini image response did not contain image bytes."
                    )
                return image_bytes, "gemini", "gemini-2.5-flash-image"
            except Exception as error:
                last_error = error
                if not _is_resource_exhausted_error(error) or attempt == 3:
                    break
        assert last_error is not None
        raise last_error

    try:
        chosen_provider = provider
        if provider == "openai":
            image_bytes, actual_provider, model_name = await generate_with_openai()
        elif provider == "gemini":
            try:
                image_bytes, actual_provider, model_name = await generate_with_gemini()
            except Exception as error:
                if _is_resource_exhausted_error(error) and os.getenv("OPENAI_API_KEY"):
                    _set_activity_status(
                        context,
                        status_key,
                        "Gemini rate-limited, falling back to OpenAI",
                    )
                    (
                        image_bytes,
                        actual_provider,
                        model_name,
                    ) = await generate_with_openai()
                else:
                    raise
        else:
            raise RuntimeError(f"Unsupported image provider: {provider}")

        destination_path.write_bytes(image_bytes)
    except Exception as error:
        _set_activity_status(
            context, status_key, f"Failed: {truncate_status(str(error))}"
        )
        return ToolExecutionResult(
            model_response={"error": f"Concept image generation failed: {error}"}
        )

    _set_activity_status(context, status_key, "Succeeded")
    return ToolExecutionResult(
        model_response={
            "result": "Concept image generated successfully.",
            "provider": actual_provider,
            "model": model_name,
            "output_path": output_relative,
        },
        metadata=ConceptImageMetadata(
            provider=actual_provider,
            model=model_name,
            output_path=output_relative,
            prompt=args.prompt,
        ),
    )


def _set_activity_status(context: AgentContext, key: str, status: str) -> None:
    context.activity_statuses[key] = status
    if context.render_activity_statuses is not None:
        context.render_activity_statuses(dict(context.activity_statuses))


async def _ensure_openscad_image(context: AgentContext) -> tuple[bool, str]:
    status_key = "docker: openscad-image"
    _set_activity_status(context, status_key, "Checking image availability")
    inspect_command = ["docker", "image", "inspect", context.openscad_image]
    inspect_returncode, _, inspect_stderr = await _run_command(
        inspect_command,
        cwd=context.workspace_root,
        timeout_seconds=DOCKER_TIMEOUT_SECONDS,
        context=context,
        status_key=status_key,
    )
    if inspect_returncode == 0:
        _set_activity_status(context, status_key, "Image ready")
        return True, ""

    build_command = [
        "docker",
        "build",
        "-t",
        context.openscad_image,
        "-f",
        "docker/openscad/Dockerfile",
        ".",
    ]
    _set_activity_status(context, status_key, "Image missing, building from Dockerfile")
    build_returncode, build_stdout, build_stderr = await _run_command(
        build_command,
        cwd=context.workspace_root,
        timeout_seconds=DOCKER_TIMEOUT_SECONDS,
        context=context,
        status_key=status_key,
    )
    if build_returncode == 0:
        details = "Docker image was missing and has been built automatically."
        if build_stdout:
            details = f"{details}\n{build_stdout}"
        _set_activity_status(context, status_key, "Image built successfully")
        return True, details

    detail_chunks = [
        f"Missing Docker image: {context.openscad_image}",
        "Tried to build it automatically with:",
        " ".join(shlex.quote(item) for item in build_command),
    ]
    if inspect_stderr:
        detail_chunks.append(f"inspect stderr:\n{inspect_stderr}")
    if build_stdout:
        detail_chunks.append(f"build stdout:\n{build_stdout}")
    if build_stderr:
        detail_chunks.append(f"build stderr:\n{build_stderr}")
    if build_returncode == 124:
        _set_activity_status(
            context,
            status_key,
            f"Build timed out after {DOCKER_TIMEOUT_SECONDS}s",
        )
    else:
        _set_activity_status(context, status_key, "Build failed")
    return False, "\n\n".join(detail_chunks)


async def _run_openscad(
    *,
    action: Literal["export_stl", "render_scad", "validate_scad"],
    input_path: str,
    output_path: str | None,
    extra_args: list[str],
    context: AgentContext,
) -> ToolExecutionResult:
    status_key = f"docker: {action} {Path(input_path).name}"
    try:
        source_path = _resolve_workspace_path(input_path, context)
        source_relative = _workspace_relative_path(source_path, context)
        if source_path.suffix.lower() != ".scad":
            return ToolExecutionResult(
                model_response={"error": "OpenSCAD tools require a .scad input file."}
            )

        destination_relative: str | None = None
        if output_path is not None:
            destination_path = _resolve_workspace_path(output_path, context)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            destination_relative = _workspace_relative_path(destination_path, context)
        else:
            destination_relative = None
    except ValueError as error:
        return ToolExecutionResult(model_response={"error": str(error)})

    image_ready, image_message = await _ensure_openscad_image(context)
    if not image_ready:
        _set_activity_status(context, status_key, "Blocked: image unavailable")
        return ToolExecutionResult(
            model_response={
                "error": f"OpenSCAD Docker image is unavailable.\n\n{image_message}",
            }
        )

    docker_command = [
        "docker",
        "run",
        "--rm",
        "--name",
        f"openscad-{action}-{uuid.uuid4().hex[:8]}",
        "-v",
        f"{context.workspace_root.resolve()}:/workspace",
        "-w",
        "/workspace",
        context.openscad_image,
    ]
    container_name = docker_command[4]

    openscad_args = list(extra_args)
    if destination_relative is None:
        openscad_args.extend(["-o", "/tmp/validate-output.stl", source_relative])
    else:
        openscad_args.extend(["-o", destination_relative, source_relative])

    if action == "render_scad":
        rendered_command = " ".join(
            shlex.quote(part) for part in ["openscad", *openscad_args]
        )
        docker_command.extend(
            [
                "sh",
                "-lc",
                (
                    "Xvfb :99 -screen 0 1024x768x24 >/tmp/xvfb.log 2>&1 & "
                    f"export DISPLAY=:99 && {rendered_command}; "
                    "code=$?; cat /tmp/xvfb.log; exit $code"
                ),
            ]
        )
    else:
        docker_command.extend(["openscad", *openscad_args])

    _set_activity_status(context, status_key, "Starting container")
    returncode, stdout, stderr = await _run_command(
        docker_command,
        cwd=context.workspace_root,
        timeout_seconds=DOCKER_TIMEOUT_SECONDS,
        context=context,
        status_key=status_key,
        cleanup_command=["docker", "rm", "-f", container_name],
    )
    command_text = " ".join(shlex.quote(item) for item in docker_command)
    if image_message:
        stdout = f"{image_message}\n\n{stdout}".strip()

    if returncode != 0:
        if returncode == 124:
            _set_activity_status(
                context,
                status_key,
                f"Timed out after {DOCKER_TIMEOUT_SECONDS}s",
            )
        else:
            _set_activity_status(
                context, status_key, f"Failed with exit code {returncode}"
            )
        return ToolExecutionResult(
            model_response={
                "error": (
                    f"OpenSCAD {action} failed with exit code {returncode}.\n"
                    f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
                ),
                "command": command_text,
                "returncode": returncode,
            },
            metadata=OpenScadMetadata(
                action=action,
                input_path=source_relative,
                output_path=destination_relative,
                command=command_text,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            ),
        )

    _set_activity_status(context, status_key, "Succeeded")
    return ToolExecutionResult(
        model_response={
            "result": f"OpenSCAD {action} succeeded.",
            "input_path": source_relative,
            "output_path": destination_relative,
            "command": command_text,
            "stdout": stdout,
            "stderr": stderr,
        },
        metadata=OpenScadMetadata(
            action=action,
            input_path=source_relative,
            output_path=destination_relative,
            command=command_text,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        ),
    )


async def validate_scad(
    args: ValidateScadArgs,
    state: RunState,
    context: AgentContext,
) -> ToolExecutionResult:
    del state
    return await _run_openscad(
        action="validate_scad",
        input_path=args.path,
        output_path=None,
        extra_args=[],
        context=context,
    )


async def export_stl(
    args: ExportStlArgs,
    state: RunState,
    context: AgentContext,
) -> ToolExecutionResult:
    del state
    if not args.output_path.lower().endswith(".stl"):
        return ToolExecutionResult(
            model_response={"error": "The STL export output_path must end with .stl."}
        )
    return await _run_openscad(
        action="export_stl",
        input_path=args.path,
        output_path=args.output_path,
        extra_args=["--export-format", "binstl"],
        context=context,
    )


async def render_scad(
    args: RenderScadArgs,
    state: RunState,
    context: AgentContext,
) -> ToolExecutionResult:
    del state
    if not args.output_path.lower().endswith(".png"):
        return ToolExecutionResult(
            model_response={"error": "The render output_path must end with .png."}
        )
    extra_args = [
        "--autocenter",
        "--viewall",
        f"--imgsize={args.width},{args.height}",
        "--render",
    ]
    if args.view is not None:
        extra_args.append(f"--view={args.view}")
    return await _run_openscad(
        action="render_scad",
        input_path=args.path,
        output_path=args.output_path,
        extra_args=extra_args,
        context=context,
    )


class GeneratePlanArgs(BaseModel):
    todos: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Call this when you have enough information from the user. "
            "Provide the initial list of todos needed to execute the task."
        ),
    )

    @field_validator("todos")
    @classmethod
    def validate_todos(cls, todos: list[str]) -> list[str]:
        normalized_todos: list[str] = []
        seen: set[str] = set()
        for todo in todos:
            normalized = " ".join(todo.split())
            if not normalized:
                raise ValueError("Todos must not be empty.")
            normalized_key = normalized.lower()
            if normalized_key in seen:
                raise ValueError("Todos must be distinct.")
            seen.add(normalized_key)
            normalized_todos.append(normalized)
        return normalized_todos


async def generate_plan(
    args: GeneratePlanArgs,
    state: RunState,
    context: AgentContext,
) -> ToolExecutionResult:
    del context
    added = state.add_todos(args.todos)
    state.mode = "execute"
    return ToolExecutionResult(
        model_response={
            "result": "Plan accepted. Start executing the design task.",
            "todos": list(state.todos),
            "mode": state.mode,
        },
        metadata=GeneratePlanMetadata(todos=added),
    )


READ_FILE_TOOL = Tool(
    name="read_file",
    description="Read a UTF-8 text file and return its contents.",
    args_model=ReadFileArgs,
    handler=read_file,
)

WRITE_FILE_TOOL = Tool(
    name="write_file",
    description="Write a UTF-8 text file to disk.",
    args_model=WriteFileArgs,
    handler=write_file,
)

EDIT_FILE_TOOL = Tool(
    name="edit_file",
    description="Replace an exact text snippet in a UTF-8 text file.",
    args_model=EditFileArgs,
    handler=edit_file,
)

MODIFY_TODO_TOOL = Tool(
    name="modify_todo",
    description="Add or remove todos from the current run state.",
    args_model=ModifyTodoArgs,
    handler=modify_todo,
)

SEARCH_WEB_TOOL = Tool(
    name="search_web",
    description="Search the web with Exa and return structured results.",
    args_model=SearchWebArgs,
    handler=search_web,
)

DELEGATE_SEARCH_TOOL = Tool(
    name="delegate_search",
    description="Delegate 1 to 3 distinct web research queries to search subagents.",
    args_model=DelegateSearchArgs,
    handler=delegate_search,
)

BASH_TOOL = Tool(
    name="bash",
    description="Run a bash command in the current working directory and capture stdout and stderr.",
    args_model=BashArgs,
    handler=bash,
)

VALIDATE_SCAD_TOOL = Tool(
    name="validate_scad",
    description="Validate an OpenSCAD source file by compiling it in Docker.",
    args_model=ValidateScadArgs,
    handler=validate_scad,
)

EXPORT_STL_TOOL = Tool(
    name="export_stl",
    description="Export an STL from a workspace OpenSCAD file using Dockerized OpenSCAD.",
    args_model=ExportStlArgs,
    handler=export_stl,
)

RENDER_SCAD_TOOL = Tool(
    name="render_scad",
    description="Render a PNG image from a workspace OpenSCAD file using Dockerized OpenSCAD.",
    args_model=RenderScadArgs,
    handler=render_scad,
)

GENERATE_PLAN_TOOL = Tool(
    name="generate_plan",
    description="Call this when you have enough information from the user.",
    args_model=GeneratePlanArgs,
    handler=generate_plan,
)

GENERATE_CONCEPT_IMAGE_TOOL = Tool(
    name="generate_concept_image",
    description="Generate a concept image for the design and save it to a workspace PNG file.",
    args_model=GenerateConceptImageArgs,
    handler=generate_concept_image,
)
