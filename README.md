# 3D Print Planning Assistant

This repo is an interactive agent for planning and iterating 3D-print designs with OpenSCAD. It keeps the planning/execution loop from the workshop on [Building your own Deep Research Agent](https://github.com/hugobowne/build-your-own-deep-research-agent) with Hugo and Ivan, but the workflow is now centered on design requirements, SCAD generation, validation, STL export, and renders.

A huge thank you to Hugo and Ivan for the inspiration around the agent runtime / harness.

## Features

- Provider-selectable CLI runtime for Gemini, OpenAI, and Anthropic
- Planning mode followed by execution mode with todo tracking
- Workspace file tools for incremental OpenSCAD edits
- Docker-backed OpenSCAD tools:
  - `validate_scad(path)`
  - `export_stl(path, output_path)`
  - `render_scad(path, output_path, ...)`
- Provider-aware concept image generation:
  - `generate_concept_image(prompt, output_path, provider=auto)`
- Generated artifacts are kept under `output/` by default
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
python app.py --provider anthropic --model claude-opus-4-6
```

`uv run` preset shortcuts:

```bash
uv run planner-openai
uv run planner-claude
uv run planner-gemini
```

They expand to:

```bash
uv run app.py --provider openai --model gpt-5.4
uv run app.py --provider anthropic --model claude-opus-4-6
uv run app.py --provider gemini --model gemini-3.1-pro-preview
```

You can still pass extra flags after the preset:

```bash
uv run planner-openai --max-iterations 20
```

Optional flags:

```bash
python app.py --provider openai --model gpt-5 --thinking-level LOW --max-iterations 30
```

Composer controls:

- `Enter` inserts a new line
- `Ctrl+J` submits the message
- Multi-line paste is supported

## Workflow

The assistant is designed to:

1. Clarify design intent, printer constraints, and dimensions.
2. Write a specification markdown file in `output/`.
3. Generate a concept image in `output/`.
4. Generate a concrete task plan.
5. Write or patch `.scad` files in `output/`.
6. Validate them with Dockerized OpenSCAD.
7. Export STL files and render PNG previews into `output/` when needed.

## Example prompt

```txt
Let's build a window cover for an interior round shaped window to be printed on an Ender 3 v2.

Here are the details:

• Window diameter: 630 mm
• Frame/trim diameter: ~640–650 mm (allowing for measurement tolerance and clearance)
• Cover style: Segmented rigid blackout disk (12 radial segments, each split into 2 printable halves with backing ring support)
• Purpose: Full light blackout for circular window (privacy + light blocking)
• Indoor/outdoor: Indoor
• Material: PETG (preferred for durability, heat resistance, and reduced brittleness vs PLA)
• Max cover diameter: 220 mm per individual printed part (Ender 3 V2 build plate constraint)
• Removable?: Yes (modular screw-assembled design with optional finger notch for removal)
• Appearance: Minimal, matte, uniform surface (front-facing smooth skin with internal ribbing hidden on rear)
```

## Notes

- OpenSCAD execution is file-first. The agent works with `.scad` files in the workspace and validates or exports them by path.
- The OpenSCAD runtime will try to build the configured Docker image automatically if it is missing.
- Search is optional. The app starts normally without Exa configured.
- CuraEngine, slicing, and post-print feedback loops are not part of this first implementation.
