<!--
  Auto-generated starter for AI coding agents.
  This file was created because no discoverable agent docs were found in the workspace.
  Update the sections below with concrete commands and examples after running a repo scan.
-->
# Copilot / AI Agent Instructions

Purpose
- Help AI coding agents become productive quickly by documenting the repo's architecture,
  developer workflows, and project-specific conventions.

Quick rules for agents
- Always scan the repository root for these files and merge any existing agent docs:
  - `README.md`, `Makefile`, `package.json`, `pyproject.toml`, `setup.cfg`, `Dockerfile`, `.github/workflows`
- If a `.github/copilot-instructions.md` already exists, merge rather than overwrite.
- When uncertain, prefer asking the human maintainer rather than guessing build or deploy commands.

Repository summary (placeholder — fill after scanning)
- Language(s): (e.g. Python, Node, Rust)
- Primary components: (list service names, CLI, web UI, libs)
- Entrypoints: (main scripts, server binaries, lambda handlers)

What to extract during your first pass
- Build & test commands: find and record exact commands for build, test, lint, and format (examples to look for: `make`, `npm test`, `pytest`, `tox`, `gradle`).
- CI/CD: list workflow files under `.github/workflows` and summarize steps (build, test, docker, deploy).
- Runtime/dependency hints: identify `requirements.txt`, `Pipfile`, `package.json`, `Cargo.toml`, or installed Docker images.
- Important configs: YAML/JSON used across services (examples: `config/*.yaml`, `env/*.env.example`).

Project-specific patterns & examples (fill with real examples)
- Example: "Services communicate via gRPC — see `services/auth/server.py` and `services/user/client.ts` for protobuf locations and client usage." 
- Example: "Single monorepo: top-level `package.json` runs `lerna` scripts; packages live under `packages/`." 
- Example: "C++ native lib: build via `scripts/build_native.sh` and `CMakeLists.txt` in `native/`." 

Conventions to preserve
- Code style and formatting: use project's formatter (add exact command, e.g. `npm run format` or `black .`).
- Branching / commit style: follow repo's CONTRIBUTING.md if present.

Merge guidance when updating this file
- Preserve any hand-authored text already in `.github/copilot-instructions.md`.
- Append an "AI-generated" section and list changes with short rationale.

If repository is empty or missing files
- Leave a minimal actionable template of commands for the maintainer to fill.
- Example placeholder commands to replace with real ones after scanning:
  - Build: `make build` or `npm run build` or `python -m build`
  - Test: `make test` or `npm test` or `pytest -q`
  - Start: `make run` or `python -m myapp` or `node ./server/index.js`

Asking for feedback
- After generating or updating this file, ask the maintainer to provide:
  1. The actual build/test/start commands.
  2. Notes about architectural decisions (why services are split, data flow, critical performance constraints).

Guide for a follow-up agent run
1. Run a file scan for files mentioned above.
2. Replace the placeholders in this file with exact commands and examples found.
3. Add 3–5 short example tasks the agent can run locally to validate changes (e.g., `make test`, `docker-compose up --build`, `pytest tests/test_x.py`).

-- End of starter template --
