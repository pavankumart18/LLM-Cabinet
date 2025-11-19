from __future__ import annotations

from dataclasses import dataclass

from .base import LlmAgent


RESEARCHER_SYSTEM = (
    "You are Researcher.\n"
    "Task: gather accurate facts, definitions, relevant background, and references\n"
    "that help solve the objective. Be concise, structured, and clear.\n"
)


ENGINEER_SYSTEM = (
    "You are Engineer.\n"
    "Task: propose a concrete, actionable solution or design. Include algorithms,\n"
    "code or pseudocode where helpful, with step-by-step reasoning and tradeoffs.\n"
)


ANALYST_SYSTEM = (
    "You are Analyst.\n"
    "Task: synthesize insights, evaluate options, highlight risks and edge cases,\n"
    "and provide clear recommendations supported by reasoning.\n"
)


SYNTHESIZER_SYSTEM = (
    "You are Synthesizer.\n"
    "Combine step outputs into a single, high-quality answer that is complete,\n"
    "cohesive, and directly addresses the user request. Structure with sections\n"
    "and numbered steps if helpful. Keep it concise and actionable.\n"
)


CRITIC_SYSTEM = (
    "You are Critic.\n"
    "Review the proposed final answer for correctness, clarity, completeness,\n"
    "and safety. Respond with STRICT JSON only:\n"
    "{\n  \"quality\": 1-5,\n  \"issues\": [\"problem description\"],\n  \"suggested_fixes\": [\"specific fix\"]\n}\n"
)


@dataclass
class ResearcherAgent(LlmAgent):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(name="researcher", system_prompt=RESEARCHER_SYSTEM, model=model)


@dataclass
class EngineerAgent(LlmAgent):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(name="engineer", system_prompt=ENGINEER_SYSTEM, model=model)


@dataclass
class AnalystAgent(LlmAgent):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(name="analyst", system_prompt=ANALYST_SYSTEM, model=model)


@dataclass
class SynthesizerAgent(LlmAgent):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(name="synthesizer", system_prompt=SYNTHESIZER_SYSTEM, model=model)


@dataclass
class CriticAgent(LlmAgent):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(name="critic", system_prompt=CRITIC_SYSTEM, model=model)

