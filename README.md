# 3D Print Planning Assistant

This repo is an interactive agent for planning and iterating 3D-print designs with OpenSCAD. It keeps the original planning/execution loop, but the workflow is now centered on design requirements, SCAD generation, validation, STL export, and renders.

## Features

- Provider-selectable CLI runtime for Gemini, OpenAI, and Anthropic
- Planning mode followed by execution mode with todo tracking
- Workspace file tools for incremental OpenSCAD edits
- Docker-backed OpenSCAD tools:
  - `validate_scad(path)`
  - `export_stl(path, output_path)`
  - `render_scad(path, output_path, ...)`
- Optional Exa-backed web search for standards or reference lookups

## Requirements

- Python 3.12+
- Docker
- A provider API key:
  - `GEMINI_API_KEY`
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`

`EXA_API_KEY` is optional and only needed if you want search tools to work.

## Build the OpenSCAD image

```bash
docker build -t 3d-print-assistant-openscad -f docker/openscad/Dockerfile .
```

Set `OPENSCAD_DOCKER_IMAGE` if you want to use a different tag.

## Run

```bash
python app.py --provider gemini --model gemini-3.1-pro
python app.py --provider openai --model gpt-5.4
python app.py --provider anthropic --model claude-opus-4.6
```

Optional flags:

```bash
python app.py --provider openai --model gpt-5 --thinking-level LOW --max-iterations 30
```

## Workflow

The assistant is designed to:

1. Clarify design intent, printer constraints, and dimensions.
2. Generate a concrete task plan.
3. Write or patch `.scad` files in the workspace.
4. Validate them with Dockerized OpenSCAD.
5. Export STL files and render PNG previews when needed.

## Notes

- OpenSCAD execution is file-first. The agent works with `.scad` files in the workspace and validates or exports them by path.
- Search is optional. The app starts normally without Exa configured.
- CuraEngine, slicing, and post-print feedback loops are not part of this first implementation.
