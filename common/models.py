from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = Field(default=None, description="Optional model override path or name")
    messages: list[Message]
    min_len: Optional[int] = None
    max_len: Optional[int] = None


class ChatResponse(BaseModel):
    text: str
    meta: dict[str, Any]
