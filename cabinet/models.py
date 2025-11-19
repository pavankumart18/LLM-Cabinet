from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List


def _load_map_from_env() -> Dict[str, str]:
    raw = os.environ.get("CABINET_MODEL_MAP")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _load_map_from_file(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@dataclass
class ModelRouter:
    default_model: str = "gpt-4o-mini"
    agent_models: Dict[str, str] = field(default_factory=dict)
    step_models: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_sources(
        cls,
        default_model: str = "gpt-4o-mini",
        file_path: Optional[str] = None,
        overrides: Optional[Dict[str, str]] = None,
    ) -> "ModelRouter":
        m: Dict[str, str] = {}
        m.update(_load_map_from_env())
        m.update(_load_map_from_file(file_path))
        if overrides:
            m.update({k: v for k, v in overrides.items() if v})
        # normalize keys to lowercase
        m = {str(k).lower(): str(v) for k, v in m.items()}
        return cls(default_model=default_model, agent_models=m)

    def set_role_map(self, mapping: Dict[str, str]) -> None:
        self.agent_models.update({str(k).lower(): str(v) for k, v in mapping.items()})

    def set_step_map(self, mapping: Dict[str, str]) -> None:
        self.step_models.update({str(k): str(v) for k, v in mapping.items()})

    def for_agent(
        self,
        agent_name: str,
        objective: str | None = None,
        guidance: str | None = None,
        step_id: Optional[str] = None,
    ) -> str:
        if step_id and step_id in self.step_models:
            return self.step_models[step_id]
        return self.agent_models.get(agent_name.lower(), self.default_model)


def load_available_models(
    inline_list: Optional[str] = None,
    file_path: Optional[str] = None,
) -> List[str]:
    # precedence: inline_list > file > env
    if inline_list:
        # allow comma-separated or JSON
        s = inline_list.strip()
        if s.startswith("["):
            try:
                arr = json.loads(s)
                return [str(x) for x in arr]
            except Exception:
                pass
        return [seg.strip() for seg in s.split(",") if seg.strip()]
    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [str(x) for x in data]
                if isinstance(data, dict) and "models" in data and isinstance(data["models"], list):
                    return [str(x) for x in data["models"]]
        except Exception:
            pass
    env_file = os.environ.get("CABINET_AVAILABLE_MODELS_FILE")
    if env_file:
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [str(x) for x in data]
                if isinstance(data, dict) and "models" in data and isinstance(data["models"], list):
                    return [str(x) for x in data["models"]]
        except Exception:
            pass
    env_val = os.environ.get("CABINET_AVAILABLE_MODELS")
    if env_val:
        env_val = env_val.strip()
        if env_val.startswith("["):
            try:
                arr = json.loads(env_val)
                return [str(x) for x in arr]
            except Exception:
                return []
        return [seg.strip() for seg in env_val.split(",") if seg.strip()]
    return []
