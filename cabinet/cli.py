import argparse
import os
import sys

from .orchestrator import Cabinet


def main(argv=None):
    argv = argv or sys.argv[1:]
    p = argparse.ArgumentParser(
        prog="cabinet",
        description="The Cabinet: multi-agent orchestration framework",
    )
    p.add_argument("question", help="User question to solve")
    p.add_argument("--model", default="gpt-4o-mini", help="Default model name")
    p.add_argument("--model-map", default=None, help="Path to JSON map of role->model (keys: planner,researcher,engineer,analyst,synthesizer,critic)")
    # Per-role overrides (take precedence over --model-map and env)
    p.add_argument("--planner-model", default=None)
    p.add_argument("--researcher-model", default=None)
    p.add_argument("--engineer-model", default=None)
    p.add_argument("--analyst-model", default=None)
    p.add_argument("--synthesizer-model", default=None)
    p.add_argument("--critic-model", default=None)
    p.add_argument("--no-parallel", action="store_true", help="Disable parallel step execution")
    p.add_argument("--iterations", type=int, default=2, help="Max critique iterations (>=1)")
    p.add_argument("--trace", action="store_true", help="Print plan and step outputs")
    # Dynamic routing inputs
    p.add_argument("--available-models", default=None, help="Comma-separated list or JSON array of allowed models")
    p.add_argument("--available-models-file", default=None, help="Path to JSON file (array or {models: [...]})")
    p.add_argument("--decider-model", default=None, help="Model used to make the routing decision")
    p.add_argument("--routing-goal", default="balanced", choices=["balanced", "quality", "speed"], help="Routing preference")
    args = p.parse_args(argv)

    if not os.environ.get("LLMFOUNDRY_TOKEN"):
        print("ERROR: LLMFOUNDRY_TOKEN is not set in environment.", file=sys.stderr)
        return 2

    overrides = {
        "planner": args.planner_model,
        "researcher": args.researcher_model,
        "engineer": args.engineer_model,
        "analyst": args.analyst_model,
        "synthesizer": args.synthesizer_model,
        "critic": args.critic_model,
    }
    # Remove None values
    overrides = {k: v for k, v in overrides.items() if v}

    from .models import ModelRouter, load_available_models

    router = ModelRouter.from_sources(
        default_model=args.model,
        file_path=args.model_map,
        overrides=overrides,
    )

    available_models = load_available_models(args.available_models, args.available_models_file)

    cabinet = Cabinet(
        default_model=router.default_model,
        model_map=router.agent_models,
        available_models=available_models,
        decider_model=args.decider_model,
        routing_goal=args.routing_goal,
    )
    result = cabinet.answer(
        args.question,
        parallel=not args.no_parallel,
        max_iterations=max(1, args.iterations),
    )

    if args.trace:
        print("\nPlan:")
        for s in result.plan.steps:
            print(f"- {s.id} [{s.agent}] {s.objective}")
        print("\nStep Outputs:")
        for sid, s in result.step_outputs.items():
            print(f"[{sid}] {s.agent}: {s.objective}\n{s.output}\n")
        if result.critique:
            print("Critique:")
            print(result.critique)
        print(f"Iterations: {result.iterations}")

    print("\nFinal Answer:\n")
    print(result.final_answer)


if __name__ == "__main__":
    raise SystemExit(main())
