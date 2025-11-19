import os
import sys

sys.path.append(".")

from cabinet import Cabinet


def main():
    if not os.environ.get("LLMFOUNDRY_TOKEN"):
        print("ERROR: LLMFOUNDRY_TOKEN is not set.")
        return 2

    question = (
        "Design a simple, scalable pipeline to classify incoming support tickets "
        "by intent and urgency, and propose an evaluation plan."
    )

    # Provide your available models once; the Decider picks role models dynamically in a single call.
    available_models = [
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

    cab = Cabinet(
        default_model=os.environ.get("CABINET_MODEL", "gpt-4o-mini"),
        available_models=available_models,
        decider_model=os.environ.get("CABINET_DECIDER_MODEL", "gpt-4o-mini"),
        routing_goal=os.environ.get("CABINET_ROUTING_GOAL", "balanced"),
    )
    result = cab.answer(question, parallel=True, max_iterations=2)

    print("Plan:")
    for s in result.plan.steps:
        print(f"- {s.id} [{s.agent}] {s.objective}")

    print("\nFinal Answer:\n")
    print(result.final_answer)


if __name__ == "__main__":
    raise SystemExit(main())
