import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from exa_py import Exa
import logfire
from rich.live import Live
from rich import box
from rich.table import Table

from agent import Agent, render_message, render_tool_call, render_tool_result
from llm import Message, MessagePart, build_model_client, get_default_model
from state import AgentContext, RunConfig, RunState
from tools import (
    BASH_TOOL,
    DELEGATE_SEARCH_TOOL,
    EDIT_FILE_TOOL,
    EXPORT_STL_TOOL,
    GENERATE_PLAN_TOOL,
    MODIFY_TODO_TOOL,
    PLAN_INSTRUCTION,
    READ_FILE_TOOL,
    RENDER_SCAD_TOOL,
    SEARCH_SUBAGENT_SYSTEM_INSTRUCTION,
    SEARCH_WEB_TOOL,
    SYSTEM_INSTRUCTION,
    VALIDATE_SCAD_TOOL,
    WRITE_FILE_TOOL,
)

logfire.configure(
    send_to_logfire="if-token-present",
    console=False,
    service_name="3d-print-assistant",
    inspect_arguments=False,
)

load_dotenv()


def truncate_cell(text: str, max_length: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 4].rstrip()}...."


def render_subagent_table(statuses: dict[str, str]) -> Table:
    table = Table(title="Subagents", box=box.SQUARE, show_lines=True)
    table.add_column("Search Query", no_wrap=False)
    table.add_column("Latest Action", no_wrap=True)
    for query, status in statuses.items():
        table.add_row(query, truncate_cell(status, 72))
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive 3D-print planning and OpenSCAD assistant."
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "gemini", "openai"],
        default="gemini",
        help="Model provider to use.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Provider-specific model ID. Defaults to a sensible model for the chosen provider.",
    )
    parser.add_argument(
        "--thinking-level",
        choices=["LOW", "MEDIUM", "HIGH"],
        default="LOW",
        help="Reasoning intensity hint passed to the model backend where supported.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=30,
        help="Maximum tool-use iterations per user turn.",
    )
    parser.add_argument(
        "--openscad-image",
        default=os.getenv("OPENSCAD_DOCKER_IMAGE", "3d-print-assistant-openscad"),
        help="Docker image tag used for OpenSCAD CLI execution.",
    )
    return parser.parse_args()


async def run_search_subagent(
    query: str,
    parent_config: RunConfig,
    context: AgentContext,
) -> dict[str, str]:
    child_config = RunConfig(
        provider=parent_config.provider,
        model=parent_config.model,
        thinking_level="LOW",
        max_iterations=2,
    )
    child_state = RunState(mode="execute")
    child_agent = Agent(
        config=child_config,
        state=child_state,
        context=context,
        plan_tools=[],
        execute_tools=[SEARCH_WEB_TOOL],
        plan_system_instruction=SEARCH_SUBAGENT_SYSTEM_INSTRUCTION,
        execute_system_instruction=SEARCH_SUBAGENT_SYSTEM_INSTRUCTION,
        span_title="agent.subagent {agent_subagent_title}",
        span_attributes={"agent_subagent_title": query},
    )

    async def update_tool_call(
        call,
        config: RunConfig,
        state: RunState,
        context: AgentContext,
    ) -> None:
        del config, state
        if call.name == "search_web" and "query" in call.args:
            context.subagent_statuses[query] = f"search_web: {call.args['query']}"
        else:
            context.subagent_statuses[query] = f"Calling {call.name}"
        if context.live is not None:
            context.live.update(render_subagent_table(context.subagent_statuses))

    child_agent.on("llm_tool_call", update_tool_call)

    child_contents = [Message(role="user", parts=[MessagePart(text=query)])]
    context.subagent_statuses[query] = "Starting"
    if context.live is not None:
        context.live.update(render_subagent_table(context.subagent_statuses))

    final_message = await child_agent.run_until_idle(child_contents)
    final_text = "\n".join(
        part.text for part in final_message.parts if part.text
    ).strip()
    context.subagent_statuses[query] = final_text[:100] or "Done"
    if context.live is not None:
        context.live.update(render_subagent_table(context.subagent_statuses))
    return {"query": query, "answer": final_text}


async def run_search_subagents(
    queries: list[str],
    parent_config: RunConfig,
    context: AgentContext,
) -> list[dict[str, str]]:
    context.subagent_statuses = {query: "Queued" for query in queries}
    with logfire.span(
        "agent.dispatch_search",
        agent_subagent_count=len(queries),
        agent_subagent_queries=queries,
    ):
        with Live(
            render_subagent_table(context.subagent_statuses), refresh_per_second=8
        ) as live:
            context.live = live
            tasks = [
                asyncio.create_task(run_search_subagent(query, parent_config, context))
                for query in queries
            ]
            results = await asyncio.gather(*tasks)
            live.update(render_subagent_table(context.subagent_statuses))
            context.live = None
            return results


async def main() -> None:
    args = parse_args()
    provider = args.provider
    model = args.model or get_default_model(provider)

    exa_api_key = os.getenv("EXA_API_KEY")
    exa = Exa(api_key=exa_api_key) if exa_api_key else None

    config = RunConfig(
        provider=provider,
        model=model,
        thinking_level=args.thinking_level,
        max_iterations=args.max_iterations,
    )
    context = AgentContext(
        exa=exa,
        model_client=build_model_client(provider),
        workspace_root=Path.cwd(),
        openscad_image=args.openscad_image,
    )
    context.search_agent_runner = lambda queries: run_search_subagents(
        queries, config, context
    )

    agent = Agent(
        config=config,
        state=RunState(mode="plan"),
        context=context,
        plan_tools=[GENERATE_PLAN_TOOL],
        execute_tools=[
            READ_FILE_TOOL,
            WRITE_FILE_TOOL,
            EDIT_FILE_TOOL,
            MODIFY_TODO_TOOL,
            DELEGATE_SEARCH_TOOL,
            SEARCH_WEB_TOOL,
            VALIDATE_SCAD_TOOL,
            EXPORT_STL_TOOL,
            RENDER_SCAD_TOOL,
            BASH_TOOL,
        ],
        plan_system_instruction=PLAN_INSTRUCTION,
        execute_system_instruction=SYSTEM_INSTRUCTION,
        span_title="agent.turn",
    )

    agent.on("message", render_message)
    agent.on("llm_tool_call", render_tool_call)
    agent.on("tool_result", render_tool_result)

    contents: list[Message] = []

    with logfire.span("agent"):
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue

            contents.append(Message(role="user", parts=[MessagePart(text=user_input)]))
            await agent.run_until_idle(contents)


if __name__ == "__main__":
    asyncio.run(main())
