from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

from ..api_client import call_llm_api


def _normalize_history(system_prompt: Optional[str], history: List[Dict[str, str]], user_content: str) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(history)
    msgs.append({"role": "user", "content": user_content})
    return msgs


@dataclass
class LlmAgent:
    name: str
    system_prompt: str
    model: str = "gpt-4o-mini"

    def run(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        model_override: Optional[str] = None,
    ) -> str:
        history = history or []
        messages = _normalize_history(self.system_prompt, history, prompt)
        selected = model_override or self.model
        return call_llm_api(selected, messages)
