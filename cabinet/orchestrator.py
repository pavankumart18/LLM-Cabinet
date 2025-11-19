from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from .blackboard import Blackboard, StepResult
from .agents import (
    PlannerAgent,
    ResearcherAgent,
    EngineerAgent,
    AnalystAgent,
    SynthesizerAgent,
    CriticAgent,
    ModelDeciderAgent,
    Plan,
    PlanStep,
)
from .models import ModelRouter
from .api_client import ModelNotFoundError, LLMAPIError


@dataclass
class CabinetResult:
    query: str
    plan: Plan
    step_outputs: Dict[str, StepResult]
    draft_answer: str
    final_answer: str
    critique: Optional[Dict[str, Any]] = None
    iterations: int = 1


class Cabinet:
    def __init__(
        self,
        default_model: str = "gpt-4o-mini",
        model_map: Optional[Dict[str, str]] = None,
        router: Optional[ModelRouter] = None,
        available_models: Optional[List[str]] = None,
        decider_model: Optional[str] = None,
        routing_goal: str = "balanced",
        max_workers: int = 4,
    ) -> None:
        # Model routing
        self.model_router = router or ModelRouter.from_sources(
            default_model=default_model,
            overrides=(model_map or {}),
        )
        self.available_models = available_models or []
        self.routing_goal = routing_goal

        # Core agents (constructed with default; overridden per-call by router)
        self.planner = PlannerAgent(model=self.model_router.default_model)
        self.researcher = ResearcherAgent(model=self.model_router.default_model)
        self.engineer = EngineerAgent(model=self.model_router.default_model)
        self.analyst = AnalystAgent(model=self.model_router.default_model)
        self.synthesizer = SynthesizerAgent(model=self.model_router.default_model)
        self.critic = CriticAgent(model=self.model_router.default_model)
        self.decider = ModelDeciderAgent(model=decider_model or self.model_router.default_model)

        self.blackboard = Blackboard()
        self.max_workers = max(1, int(max_workers))

        self._agent_map = {
            "researcher": self.researcher,
            "engineer": self.engineer,
            "analyst": self.analyst,
        }

    def _run_step(self, step: PlanStep, query: str) -> StepResult:
        agent = self._agent_map.get(step.agent, self.researcher)
        prompt = (
            f"User request: {query}\n\n"
            f"Your step ({step.id} - {step.agent}): {step.objective}\n"
            f"Guidance: {step.guidance}"
        )
        primary = self.model_router.for_agent(step.agent, step.objective, step.guidance, step_id=step.id)
        candidates = [
            primary,
            self.model_router.default_model,
            "gpt-4o-mini",
            "claude-3-haiku-20240307",
            "gemini-1.5-flash-8b",
        ]
        output = self._try_run(agent, prompt, candidates)
        return StepResult(step_id=step.id, agent=step.agent, objective=step.objective, output=output)

    @staticmethod
    def _steps_context_text(step_outputs: Dict[str, StepResult]) -> str:
        lines: List[str] = []
        for sid in sorted(step_outputs.keys()):
            s = step_outputs[sid]
            lines.append(f"[{sid}] {s.agent} — {s.objective}\n{s.output}\n")
        return "\n".join(lines)

    @staticmethod
    def _parse_critic_json(text: str) -> Dict[str, Any]:
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
                    return {"quality": 3, "issues": [], "suggested_fixes": []}
        return {"quality": 3, "issues": [], "suggested_fixes": []}

    def _try_run(self, agent, prompt: str, candidates: List[str]) -> str:
        last_err: Optional[Exception] = None
        for m in [c for c in candidates if c]:
            try:
                return agent.run(prompt, model_override=m)
            except ModelNotFoundError as e:
                last_err = e
                continue
            except LLMAPIError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue
        if last_err:
            raise last_err
        raise RuntimeError("No model candidates provided")

    def answer(
        self,
        query: str,
        parallel: bool = True,
        max_iterations: int = 2,
    ) -> CabinetResult:
        # 0) Decide model routing (single call) if available models provided
        if self.available_models:
            # Try decider with its configured model; if invalid, fall back to safer choices.
            decider_overrides = [
                getattr(self.decider, "model", None),
                "gpt-4o-mini",
                "claude-3-haiku-20240307",
                "gemini-1.5-flash-8b",
                self.model_router.default_model,
            ]
            decider_overrides = [m for m in decider_overrides if m]
            decision = None
            last_error: Optional[Exception] = None
            for m in decider_overrides:
                try:
                    decision = self.decider.decide(
                        user_request=query,
                        allowed_models=self.available_models,
                        routing_goal=self.routing_goal,
                        model_override=m,
                    )
                    break
                except ModelNotFoundError:
                    last_error = None
                    continue
                except LLMAPIError as e:
                    last_error = e
                    continue
                except Exception as e:
                    last_error = e
                    continue
            if isinstance(decision, dict):
                role_map = decision.get("role_models") or {}
                if isinstance(role_map, dict) and role_map:
                    self.model_router.set_role_map(role_map)
            # If still no decision, proceed with existing router map

        # 1) Plan
        plan_model = self.model_router.for_agent("planner", query)
        plan_candidates = [
            plan_model,
            self.model_router.default_model,
            "gpt-4o-mini",
            "claude-3-haiku-20240307",
            "gemini-1.5-flash-8b",
        ]
        plan = None
        last_err: Optional[Exception] = None
        for m in [c for c in plan_candidates if c]:
            try:
                plan = self.planner.plan(query, model_override=m)
                break
            except ModelNotFoundError as e:
                last_err = e
                continue
            except LLMAPIError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue
        if plan is None:
            if last_err:
                raise last_err
            raise RuntimeError("Planner could not be run with any candidate model")

        # 2) Execute steps
        step_outputs: Dict[str, StepResult] = {}
        if parallel and len(plan.steps) > 1:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(plan.steps))) as ex:
                future_map = {ex.submit(self._run_step, step, query): step.id for step in plan.steps}
                for fut in as_completed(future_map):
                    res = fut.result()
                    step_outputs[res.step_id] = res
                    self.blackboard.record_step(res)
        else:
            for step in plan.steps:
                res = self._run_step(step, query)
                step_outputs[res.step_id] = res
                self.blackboard.record_step(res)

        # 3) Synthesize
        context_text = self._steps_context_text(step_outputs)
        synth_prompt = (
            f"User request: {query}\n\n"
            f"Context from team steps below. Produce a cohesive final answer.\n\n"
            f"{context_text}"
        )
        synth_model = self.model_router.for_agent("synthesizer")
        synth_candidates = [
            synth_model,
            self.model_router.default_model,
            "gpt-4o-mini",
            "claude-3-haiku-20240307",
            "gemini-1.5-flash-8b",
        ]
        draft_answer = self._try_run(self.synthesizer, synth_prompt, synth_candidates)

        # 4) Critique & iterate
        final_answer = draft_answer
        critique_dict: Optional[Dict[str, Any]] = None
        iterations = 1
        for i in range(max_iterations - 1):
            crit_prompt = (
                f"User request: {query}\n\n"
                f"Proposed final answer:\n{final_answer}\n\n"
                f"Team context:\n{context_text}"
            )
            critic_model = self.model_router.for_agent("critic")
            critic_candidates = [
                critic_model,
                self.model_router.default_model,
                "gpt-4o-mini",
                "claude-3-haiku-20240307",
                "gemini-1.5-flash-8b",
            ]
            critique_text = self._try_run(self.critic, crit_prompt, critic_candidates)
            critique = self._parse_critic_json(critique_text)
            critique_dict = critique
            issues = critique.get("issues", []) or []
            quality = critique.get("quality", 3)

            if not issues and quality >= 4:
                break

            fix_prompt = (
                f"User request: {query}\n\n"
                f"Improve the final answer based on this critique:\n"
                f"quality={quality}, issues={issues}, suggested_fixes={critique.get('suggested_fixes', [])}\n\n"
                f"Current answer:\n{final_answer}"
            )
            final_answer = self._try_run(self.synthesizer, fix_prompt, synth_candidates)
            iterations += 1

        return CabinetResult(
            query=query,
            plan=plan,
            step_outputs=step_outputs,
            draft_answer=draft_answer,
            final_answer=final_answer,
            critique=critique_dict,
            iterations=iterations,
        )
