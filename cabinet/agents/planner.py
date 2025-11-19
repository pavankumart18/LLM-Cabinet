from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import json

from .base import LlmAgent


PLANNER_SYSTEM_PROMPT = (
    "You are Planner, coordinating a team of specialist agents (researcher, engineer, analyst).\n"
    "Given a complex user request, produce a crisp plan with 2-6 steps,\n"
    "each assigned to an agent type.\n"
    "Output STRICT JSON only with the schema:\n"
    "{\n  \"steps\": [\n    {\n      \"id\": \"s1\",\n      \"agent\": \"researcher|engineer|analyst\",\n      \"objective\": \"short goal\",\n      \"guidance\": \"specific tips\"\n    }\n  ]\n}\n"
    "No prose outside JSON. Favor minimal, actionable steps."
)


@dataclass
class PlanStep:
    id: str
    agent: str
    objective: str
    guidance: str


@dataclass
class Plan:
    steps: List[PlanStep]


class PlannerAgent(LlmAgent):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(name="planner", system_prompt=PLANNER_SYSTEM_PROMPT, model=model)

    def plan(self, user_request: str, model_override: str | None = None) -> Plan:
        raw = self.run(user_request, model_override=model_override)
        data = self._parse_json(raw)
        steps = [
            PlanStep(
                id=s.get("id") or f"s{i+1}",
                agent=(s.get("agent") or "researcher").lower(),
                objective=s.get("objective") or "",
                guidance=s.get("guidance") or "",
            )
            for i, s in enumerate(data.get("steps", []))
        ]
        if not steps:
            # Fallback minimal plan
            steps = [
                PlanStep(id="s1", agent="researcher", objective="gather facts and definitions", guidance=""),
                PlanStep(id="s2", agent="engineer", objective="propose approach and solution", guidance=""),
                PlanStep(id="s3", agent="analyst", objective="analyze tradeoffs and edge cases", guidance=""),
            ]
        return Plan(steps=steps)

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        # Try strict JSON first
        try:
            return json.loads(text)
        except Exception:
            pass
        # Try to extract JSON object boundaries
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                return {"steps": []}
        return {"steps": []}
