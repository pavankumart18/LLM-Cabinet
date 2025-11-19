from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Message:
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatHistory:
    system_prompt: str | None = None
    messages: List[Message] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))

    def as_openai(self) -> List[Dict[str, str]]:
        data: List[Dict[str, str]] = []
        if self.system_prompt:
            data.append({"role": "system", "content": self.system_prompt})
        data.extend([m.to_dict() for m in self.messages])
        return data

    def copy(self) -> "ChatHistory":
        ch = ChatHistory(system_prompt=self.system_prompt)
        ch.messages = list(self.messages)
        return ch

