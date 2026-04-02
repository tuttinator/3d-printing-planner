import os
import subprocess
import sys


def _run_with_preset(*, provider: str, model: str) -> int:
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    command = [
        sys.executable,
        app_path,
        "--provider",
        provider,
        "--model",
        model,
        *sys.argv[1:],
    ]
    return subprocess.call(command)


def openai_main() -> None:
    raise SystemExit(_run_with_preset(provider="openai", model="gpt-5.4"))


def claude_main() -> None:
    raise SystemExit(_run_with_preset(provider="anthropic", model="claude-opus-4-6"))


def gemini_main() -> None:
    raise SystemExit(
        _run_with_preset(provider="gemini", model="gemini-3.1-pro-preview")
    )
