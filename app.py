import argparse
import asyncio
import os
from pathlib import Path

import logfire
from dotenv import load_dotenv
from exa_py import Exa
from rich import box
from rich.markdown import Markdown
from rich.table import Table

from agent import Agent, clip_text
from llm import Message, MessagePart, build_model_client, get_default_model
from prompts import (
    PLAN_INSTRUCTION,
    SEARCH_SUBAGENT_SYSTEM_INSTRUCTION,
    SYSTEM_INSTRUCTION,
)
from shell import Shell
from state import AgentContext, RunConfig, RunState
from tools import (
    BASH_TOOL,
    DELEGATE_SEARCH_TOOL,
    EDIT_FILE_TOOL,
    EXPORT_STL_TOOL,
    GENERATE_CONCEPT_IMAGE_TOOL,
    GENERATE_PLAN_TOOL,
    MODIFY_TODO_TOOL,
    READ_FILE_TOOL,
    RENDER_SCAD_TOOL,
    SEARCH_WEB_TOOL,
    VALIDATE_SCAD_TOOL,
    WRITE_FILE_TOOL,
    BashMetadata,
    ConceptImageMetadata,
    EditFileMetadata,
    GeneratePlanMetadata,
    OpenScadMetadata,
    ReadFileMetadata,
    WriteFileMetadata,
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


def render_status_table(statuses: dict[str, str]) -> Table:
    table = Table(title="Subagents / Jobs", box=box.SQUARE, show_lines=True)
    table.add_column("Task", no_wrap=False)
    table.add_column("Latest Action", no_wrap=True)
    if not statuses:
        table.add_row("Idle", "No active background jobs")
        return table
    for name, status in statuses.items():
        table.add_row(name, truncate_cell(status, 72))
    return table


def render_todo_table(todos: list[str]) -> Table:
    table = Table(title="TODOs", box=box.SQUARE, show_lines=True)
    table.add_column("Pending Work", no_wrap=False)
    if not todos:
        table.add_row("No pending todos")
        return table
    for todo in todos:
        table.add_row(todo)
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
    status_key = f"search: {query}"
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
            context.activity_statuses[status_key] = f"search_web: {call.args['query']}"
        else:
            context.activity_statuses[status_key] = f"Calling {call.name}"
        if context.render_activity_statuses is not None:
            context.render_activity_statuses(dict(context.activity_statuses))

    child_agent.on("llm_tool_call", update_tool_call)

    child_contents = [Message(role="user", parts=[MessagePart(text=query)])]
    context.activity_statuses[status_key] = "Starting"
    if context.render_activity_statuses is not None:
        context.render_activity_statuses(dict(context.activity_statuses))

    final_message = await child_agent.run_until_idle(child_contents)
    final_text = "\n".join(
        part.text for part in final_message.parts if part.text
    ).strip()
    context.activity_statuses[status_key] = final_text[:100] or "Done"
    if context.render_activity_statuses is not None:
        context.render_activity_statuses(dict(context.activity_statuses))
    return {"query": query, "answer": final_text}


async def run_search_subagents(
    queries: list[str],
    parent_config: RunConfig,
    context: AgentContext,
) -> list[dict[str, str]]:
    context.activity_statuses = {
        **{
            key: value
            for key, value in context.activity_statuses.items()
            if not key.startswith("search:")
        },
        **{f"search: {query}": "Queued" for query in queries},
    }
    if context.render_activity_statuses is not None:
        context.render_activity_statuses(dict(context.activity_statuses))
    with logfire.span(
        "agent.dispatch_search",
        agent_subagent_count=len(queries),
        agent_subagent_queries=queries,
    ):
        tasks = [
            asyncio.create_task(run_search_subagent(query, parent_config, context))
            for query in queries
        ]
        results = await asyncio.gather(*tasks)
        if context.render_activity_statuses is not None:
            context.render_activity_statuses(dict(context.activity_statuses))
        return results


def main() -> None:
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
        preferred_image_provider=(
            provider if provider in {"gemini", "openai"} else "auto"
        ),
    )
    context.search_agent_runner = lambda queries: run_search_subagents(
        queries, config, context
    )

    agent = Agent(
        config=config,
        state=RunState(mode="plan"),
        context=context,
        plan_tools=[
            GENERATE_PLAN_TOOL,
            DELEGATE_SEARCH_TOOL,
            SEARCH_WEB_TOOL,
        ],
        execute_tools=[
            READ_FILE_TOOL,
            WRITE_FILE_TOOL,
            EDIT_FILE_TOOL,
            MODIFY_TODO_TOOL,
            GENERATE_CONCEPT_IMAGE_TOOL,
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

    contents: list[Message] = []
    shell = Shell()

    def update_todos_region(todos: list[str]) -> None:
        shell.update_region("todos", render_todo_table(todos))

    def update_jobs_region(statuses: dict[str, str]) -> None:
        shell.update_region("jobs", render_status_table(statuses))

    context.render_todos = update_todos_region
    context.render_activity_statuses = update_jobs_region

    async def render_message_to_shell(
        message: Message,
        config: RunConfig,
        state: RunState,
        context: AgentContext,
    ) -> None:
        del config, state, context
        for part in message.parts:
            if part.text:
                shell.print(Markdown(part.text))

    async def render_tool_call_to_shell(
        call,
        config: RunConfig,
        state: RunState,
        context: AgentContext,
    ) -> None:
        del call, config, state, context
        return

    async def render_tool_result_to_shell(
        call,
        result,
        config: RunConfig,
        state: RunState,
        context: AgentContext,
    ) -> None:
        del call, config, state, context
        error = result.model_response.get("error")
        if error:
            shell.print(Markdown(f"```text\n{error}\n```"))
            return

        metadata = result.metadata
        if isinstance(metadata, ReadFileMetadata):
            shell.print(f"Read file: {metadata.path}")
            shell.print(Markdown(f"```text\n{clip_text(metadata.contents)}\n```"))
        if isinstance(metadata, WriteFileMetadata):
            shell.print(f"Wrote file: {metadata.path}")
        if isinstance(metadata, EditFileMetadata):
            shell.print(f"Edited file: {metadata.path}")
        if isinstance(metadata, BashMetadata):
            shell.print(f"Ran command: {metadata.command}")
            shell.print(Markdown(f"```text\n{clip_text(metadata.stdout)}\n```"))
            if metadata.stderr:
                shell.print(Markdown(f"```text\n{clip_text(metadata.stderr)}\n```"))
        if isinstance(metadata, OpenScadMetadata):
            shell.print(f"OpenSCAD {metadata.action}: {metadata.input_path}")
            if metadata.output_path:
                shell.print(f"Output: {metadata.output_path}")
            if metadata.stdout:
                shell.print(Markdown(f"```text\n{clip_text(metadata.stdout)}\n```"))
            if metadata.stderr:
                shell.print(Markdown(f"```text\n{clip_text(metadata.stderr)}\n```"))
        if isinstance(metadata, ConceptImageMetadata):
            shell.print(
                f"Concept image generated: {metadata.output_path} "
                f"({metadata.provider}, {metadata.model})"
            )
        if isinstance(metadata, GeneratePlanMetadata):
            shell.print("[green]Switched to execute mode.[/green]")

    agent.on("message", render_message_to_shell)
    agent.on("llm_tool_call", render_tool_call_to_shell)
    agent.on("tool_result", render_tool_result_to_shell)

    async def on_submit(text: str) -> None:
        shell.print(Markdown(f"```text\n{text}\n```"))
        shell.set_loading(True)
        try:
            contents.append(Message(role="user", parts=[MessagePart(text=text)]))
            await agent.run_until_idle(contents)
        finally:
            shell.set_loading(False)

    async def on_ready() -> None:
        update_todos_region(list(agent.state.todos))
        update_jobs_region(dict(context.activity_statuses))
        shell.print(
            "3D Print Planning Assistant\n"
            "Enter adds a new line. Ctrl+J submits. Multi-line paste is supported."
        )

    with logfire.span("agent"):
        shell.initialize(on_submit=on_submit, on_ready=on_ready)
        shell.run()


if __name__ == "__main__":
    main()
