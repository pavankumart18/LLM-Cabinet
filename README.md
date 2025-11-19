The Cabinet — Multi‑Agent Orchestration
======================================

Overview
--------
- The Cabinet coordinates multiple LLM agents (Planner, Researcher, Engineer, Analyst, Synthesizer, Critic) to tackle complex queries.
- Uses the provided LLM Foundry Chat Completions API via `requests` only.

Quick Start
-----------
- Prerequisite: set `LLMFOUNDRY_TOKEN` in your environment.
  - Windows (PowerShell): `$Env:LLMFOUNDRY_TOKEN = "<your_token>"`
  - macOS/Linux: `export LLMFOUNDRY_TOKEN="<your_token>"`

- Easiest way (single command)
  - `uv run ask.py "Your question here"`
  - Everything else is dynamic: a one-call Decider picks models per role from a built-in list (you can edit `ask.py` to change the list anytime). Optionally set env `CABINET_TRACE=1` for a brief plan + routing printout.
  - Rate limits: The client auto-retries on 429/5xx with exponential backoff. By default, `ask.py` runs steps sequentially to be friendly to quotas. To enable parallelism: `uv run ask.py "..." CABINET_PARALLEL=1 CABINET_MAX_WORKERS=2`.

- CLI
  - Default model for all roles:
    - `python -m cabinet.cli "Your question here" --model gpt-4o-mini --iterations 2 --trace`
  - Per-role model routing via JSON file:
    - Create `models.json` with keys: `planner`, `researcher`, `engineer`, `analyst`, `synthesizer`, `critic`.
    - `python -m cabinet.cli "Your question here" --model-map models.json`
  - Per-role flags (override file/env):
    - `--planner-model`, `--researcher-model`, `--engineer-model`, `--analyst-model`, `--synthesizer-model`, `--critic-model`.

- Example script
  - `python examples/ask_cabinet.py`

Design
------
- `PlannerAgent` creates a JSON plan of steps assigned to agent types.
- Steps execute (optionally in parallel) by specialist agents.
- `SynthesizerAgent` composes a cohesive draft answer.
- `CriticAgent` reviews with JSON feedback; the system may iterate to improve the answer.
- `ModelDeciderAgent` (one-call router) selects models per role from your allowed list before planning.

Key Files
---------
- `cabinet/api_client.py` — Required API client using `requests` only.
- `cabinet/models.py` — Simple `ModelRouter` for per-role routing.
- `cabinet/agents/*` — Base agent + Planner, Specialists, Synthesizer, Critic.
- `cabinet/agents/decider.py` — Decider that assigns models to roles in one call.
- `cabinet/orchestrator.py` — `Cabinet` class to orchestrate planning, execution, synthesis, critique.
- `cabinet/cli.py` — Simple CLI.

Notes
-----
- The planner and critic expect strict JSON. If parsing fails, The Cabinet falls back to sensible defaults.
- Static routing: Use `--model-map`, per-role flags, or env `CABINET_MODEL_MAP` (JSON string).
- Dynamic routing (one-call): Provide allowed models via `--available-models`/`--available-models-file` or env `CABINET_AVAILABLE_MODELS`. Optionally set `--decider-model` and `--routing-goal` (balanced|quality|speed). The Decider makes a single call to pick models for roles, then the system proceeds with that mapping.
- Rate limiting: The API client retries 429/5xx with exponential backoff. Configure via env `CABINET_API_MAX_RETRIES` (default 5) and `CABINET_API_BACKOFF` seconds (default 1.0).
