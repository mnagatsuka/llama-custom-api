from __future__ import annotations

from fastapi import APIRouter, HTTPException

from common.models import ChatRequest, ChatResponse, Message
from ..engine import generate
from common.config import get_settings


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    settings = get_settings()
    min_len = settings.min_len
    max_len = settings.max_len
    if min_len > max_len:
        raise HTTPException(status_code=500, detail="Server misconfiguration: MIN_LEN > MAX_LEN")

    # Enforce constant system prompt and fixed length window
    base_messages: list[Message] = [Message(role="system", content=settings.system_prompt)]
    base_messages += [m for m in req.messages if m.role != "system"]

    result = generate(
        messages=[m.model_dump() for m in base_messages],
        min_len=min_len,
        max_len=max_len,
        model_override=req.model,
    )
    return ChatResponse(**result)
