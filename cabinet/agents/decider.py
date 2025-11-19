from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Any, List

from .base import LlmAgent


DECIDER_SYSTEM = (
    "You are Model Decider.\n"
    "Given a user request and a list of allowed models, choose the best model\n"
    "for each agent role in this system: planner, researcher, engineer, analyst, synthesizer, critic.\n"
    "Consider routing_goal (balanced|quality|speed), task characteristics, and model strengths.\n"
    "Output STRICT JSON only with schema:\n"
    "{\n  \"role_models\": {\n    \"planner\": \"model-name\",\n    \"researcher\": \"model-name\",\n    \"engineer\": \"model-name\",\n    \"analyst\": \"model-name\",\n    \"synthesizer\": \"model-name\",\n    \"critic\": \"model-name\"\n  },\n  \"rationale\": \"1-2 sentences\"\n}\n"
    "Only pick from the provided allowed_models. Include every role in role_models."
)


@dataclass
class ModelDeciderAgent(LlmAgent):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(name="decider", system_prompt=DECIDER_SYSTEM, model=model)

    def decide(self, user_request: str, allowed_models: List[str], routing_goal: str = "balanced", model_override: str | None = None) -> Dict[str, Any]:
        prompt = (
            "Routing goal: "
            + routing_goal
            + "\nAllowed models (JSON array):\n"
            + json.dumps(allowed_models)
            + "\nUser request:\n"
            + user_request
        )
        raw = self.run(prompt, model_override=model_override)
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    return {"role_models": {}, "rationale": ""}
        return {"role_models": {}, "rationale": ""}

