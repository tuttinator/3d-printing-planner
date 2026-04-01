from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal, TypeAlias

from exa_py import Exa

if TYPE_CHECKING:
    from llm import ModelClient

SearchAgentRunner = Callable[[list[str]], Awaitable[list[dict[str, str]]]]
StatusRenderer = Callable[[dict[str, str]], None]
TodoRenderer = Callable[[list[str]], None]
Mode: TypeAlias = Literal["plan", "execute"]
Provider: TypeAlias = Literal["anthropic", "gemini", "openai"]
ImageProvider: TypeAlias = Literal["auto", "gemini", "openai"]


@dataclass(slots=True)
class RunConfig:
    provider: Provider = "gemini"
    model: str = "gemini-2.5-pro"
    thinking_level: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    max_iterations: int = 30

    def telemetry_attributes(self) -> dict[str, str | int]:
        return {
            "agent.config.provider": self.provider,
            "agent.config.model": self.model,
            "agent.config.thinking_level": self.thinking_level,
            "agent.config.max_iterations": self.max_iterations,
        }


@dataclass(slots=True)
class RunState:
    iteration_count: int = 0
    mode: Mode = "plan"
    todos: list[str] = field(default_factory=list)

    def telemetry_attributes(self) -> dict[str, str | int | list[str]]:
        return {
            "agent.state.iteration_count": self.iteration_count,
            "agent.state.mode": self.mode,
            "agent.state.todo_count": len(self.todos),
            "agent.state.todos": list(self.todos),
        }

    def add_todos(self, todos: list[str]) -> list[str]:
        added: list[str] = []
        for todo in todos:
            todo = todo.strip()
            if todo and todo not in self.todos:
                self.todos.append(todo)
                added.append(todo)
        return added

    def remove_todos(self, todos: list[str]) -> tuple[list[str], list[str]]:
        removed: list[str] = []
        not_found: list[str] = []
        for todo in todos:
            todo = todo.strip()
            todo_lower = todo.lower()
            existing = next(
                (item for item in self.todos if item.lower() == todo_lower),
                None,
            )
            if existing is None:
                not_found.append(todo)
                continue
            self.todos.remove(existing)
            removed.append(existing)
        return removed, not_found

    def is_incomplete(self) -> str | None:
        if self.mode == "plan":
            return None

        if self.todos:
            return f"""
You still have pending todos.

<todos>
{chr(10).join(self.todos)}
</todos>

Check off all todos before you end.
""".strip()

        return None


@dataclass(slots=True)
class AgentContext:
    exa: Exa | None = None
    model_client: "ModelClient | None" = None
    search_agent_runner: SearchAgentRunner | None = None
    live: Any | None = None
    activity_statuses: dict[str, str] = field(default_factory=dict)
    render_activity_statuses: StatusRenderer | None = None
    render_todos: TodoRenderer | None = None
    workspace_root: Path = field(default_factory=Path.cwd)
    openscad_image: str = "3d-print-assistant-openscad"
    preferred_image_provider: ImageProvider = "auto"
