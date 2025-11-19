import os
import sys

from cabinet import Cabinet
from cabinet.models import load_available_models


def pick_decider_model(allowed: list[str]) -> str:
    # Prefer stronger models for better routing when available
    preference = [
        "gpt-4o-mini",
        "claude-3-haiku-20240307",
        "gemini-1.5-flash-8b",
        "llama-3.3-70b-versatile",
        "qwen/qwen-2-vl-72b-instruct",
        "nousresearch/hermes-3-llama-3.1-405b",
        "microsoft/phi-3.5-mini-128k-instruct",
        "meta-llama/llama-3.2-11b-vision-instruct",
        "mistralai/pixtral-12b",
    ]
    for m in preference:
        if m in allowed:
            return m
    return allowed[0] if allowed else "gpt-4o-mini"


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv:
        print("Usage: uv run ask.py \"Your question here\"")
        return 2

    # Convenience: treat trailing TOKENS of form KEY=VALUE as env overrides (e.g., CABINET_TRACE=1)
    question_tokens: list[str] = []
    for tok in argv:
        if (
            "=" in tok
            and tok.split("=", 1)[0].isidentifier()
            and tok.split("=", 1)[0].upper() == tok.split("=", 1)[0]
        ):
            k, v = tok.split("=", 1)
            os.environ[k] = v
        else:
            question_tokens.append(tok)

    question = " ".join(question_tokens).strip()

    if not os.environ.get("LLMFOUNDRY_TOKEN"):
        print("ERROR: LLMFOUNDRY_TOKEN is not set in the environment.")
        return 2

    # Your fixed available models (edit here or override via env `CABINET_AVAILABLE_MODELS` or file `CABINET_AVAILABLE_MODELS_FILE`)
    available_models = load_available_models() or [
        "gpt-4o-mini",
        "claude-3-haiku-20240307",
        "gemini-1.5-flash-8b",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-3.2-11b-vision-instruct",
        "mistralai/pixtral-12b",
        "qwen/qwen-2-vl-72b-instruct",
        "nousresearch/hermes-3-llama-3.1-405b",
        "microsoft/phi-3.5-mini-128k-instruct",
    ]

    default_model = os.environ.get("CABINET_MODEL", "gpt-4o-mini")
    decider_model = os.environ.get("CABINET_DECIDER_MODEL", pick_decider_model(available_models))
    routing_goal = os.environ.get("CABINET_ROUTING_GOAL", "balanced")

    max_workers = int(os.environ.get("CABINET_MAX_WORKERS", "2"))

    cabinet = Cabinet(
        default_model=default_model,
        available_models=available_models,
        decider_model=decider_model,
        routing_goal=routing_goal,
        max_workers=max_workers,
    )

    parallel_env = os.environ.get("CABINET_PARALLEL", "0").lower()
    parallel = parallel_env in ("1", "true", "yes", "on")
    max_iterations = int(os.environ.get("CABINET_ITERATIONS", "2"))

    result = cabinet.answer(question, parallel=parallel, max_iterations=max_iterations)

    # Optional minimal trace if requested via env
    if os.environ.get("CABINET_TRACE") == "1":
        print("Plan:")
        for s in result.plan.steps:
            print(f"- {s.id} [{s.agent}] {s.objective}")
        print("\nRouting (role -> model):")
        for role, model in cabinet.model_router.agent_models.items():
            print(f"- {role}: {model}")
        if result.critique:
            print("\nCritique:")
            print(result.critique)
        print(f"Iterations: {result.iterations}")

    print(result.final_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
