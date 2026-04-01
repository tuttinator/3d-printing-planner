import asyncio
import shlex
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, Field, field_validator

from state import AgentContext, RunState


ArgsT = TypeVar("ArgsT", bound=BaseModel)
MAX_DELEGATED_QUERIES = 3
TODAY = datetime.now().strftime("%d %B %Y")


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


ToolMetadata: TypeAlias = (
    BashMetadata
    | DelegateSearchMetadata
    | EditFileMetadata
    | GeneratePlanMetadata
    | ModifyTodoMetadata
    | OpenScadMetadata
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


async def _run_openscad(
    *,
    action: Literal["export_stl", "render_scad", "validate_scad"],
    input_path: str,
    output_path: str | None,
    extra_args: list[str],
    context: AgentContext,
) -> ToolExecutionResult:
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

    docker_command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{context.workspace_root.resolve()}:/workspace",
        "-w",
        "/workspace",
        context.openscad_image,
        "xvfb-run",
        "-a",
        "openscad",
        *extra_args,
    ]

    if destination_relative is None:
        docker_command.extend(["-o", "/tmp/validate-output.stl", source_relative])
    else:
        docker_command.extend(["-o", destination_relative, source_relative])

    process = await asyncio.create_subprocess_exec(
        *docker_command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await process.communicate()
    stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
    returncode = process.returncode or 0
    command_text = " ".join(shlex.quote(item) for item in docker_command)

    if returncode != 0:
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


PLAN_INSTRUCTION = """
You are Koroku, a 3D printing design-planning assistant.
Today's date is {today}.

When the user asks for a part, assembly, or design analysis task:
- clarify the design intent, dimensions, printer constraints, material, and success criteria
- ask concise follow-up questions only when they materially affect the design
- generate an execution plan once enough detail is available

The execution plan should focus on practical CAD and validation work, not research writing.
Use the generate_plan tool when you have enough information to start.
Keep todos concrete and implementation-oriented.
""".strip().format(today=TODAY)

SYSTEM_INSTRUCTION = """
You are Kuroko, a 3D printing design assistant. Today's date is {today}.

Your job is to help the user plan, generate, inspect, and iterate OpenSCAD-based 3D-print designs.

Core behavior:
1. Use modify_todo to track design steps and validation work.
2. Clarify missing requirements when they affect geometry, fit, printer compatibility, or printability.
3. Write and patch OpenSCAD files incrementally using write_file and edit_file.
4. Validate OpenSCAD before claiming a design iteration is good.
5. Use validate_scad after significant SCAD edits.
6. Use export_stl and render_scad when you need artifacts for validation or delivery.
7. Keep outputs grounded in printer constraints, wall thickness, tolerances, and manufacturability.
8. Use bash only for local inspection or non-destructive checks.
9. If web search is helpful for a standard, spec, or part reference, use delegate_search or search_web, but do not depend on it by default.
10. Do not produce long research reports. Focus on design decisions, CAD changes, validation results, and next actions.

Execution style:
- Work iteratively.
- Prefer minimal SCAD patches instead of full rewrites when a file already exists.
- Surface validation failures clearly and fix them before moving on.
- If a design assumption is necessary, state it briefly and proceed.
""".strip().format(today=TODAY)

SEARCH_SUBAGENT_SYSTEM_INSTRUCTION = """
You are a focused web research subagent supporting a 3D-printing design assistant.
Today's date is {today}.
Answer the user's query in concise natural language.
Use the search_web tool when needed.
Do not ask follow-up questions.
Include source URLs in your final response when you make factual claims.
""".strip().format(today=TODAY)
