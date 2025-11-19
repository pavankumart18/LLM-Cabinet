from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class StepResult:
    step_id: str
    agent: str
    objective: str
    output: str


@dataclass
class Blackboard:
    notes: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    steps: Dict[str, StepResult] = field(default_factory=dict)

    def add_note(self, text: str) -> None:
        self.notes.append(text)

    def add_artifact(self, name: str, value: Any) -> None:
        self.artifacts[name] = value

    def record_step(self, result: StepResult) -> None:
        self.steps[result.step_id] = result

    def summarize(self) -> str:
        lines: List[str] = []
        if self.notes:
            lines.append("Notes:")
            for n in self.notes:
                lines.append(f"- {n}")
        if self.artifacts:
            lines.append("Artifacts:")
            for k in self.artifacts:
                lines.append(f"- {k}")
        if self.steps:
            lines.append("Step Outputs:")
            for sid in self.steps:
                s = self.steps[sid]
                lines.append(f"[{sid}] {s.agent}: {s.objective}")
        return "\n".join(lines)

