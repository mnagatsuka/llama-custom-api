from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

from common.config import get_settings
from common.utils.logging import get_logger
from common.utils.text_sanitize import auto_close_pairs, safe_trim
from common.inference.tokenizer import count_chars

logger = get_logger(__name__)

_llama_lock = threading.Lock()
_llama: Any = None


def _ensure_llama(model_path: Optional[str] = None) -> Any:
    global _llama
    if _llama is not None:
        return _llama
    with _llama_lock:
        if _llama is None:
            from llama_cpp import Llama  # type: ignore

            settings = get_settings()
            _llama = Llama(
                model_path=model_path or settings.model_path,
                n_ctx=settings.ctx_size,
                n_threads=settings.n_threads,
                verbose=False,
            )
    return _llama


def _build_prompt(messages: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"[system]\n{content}\n")
        elif role == "assistant":
            parts.append(f"[assistant]\n{content}\n")
        else:
            parts.append(f"[user]\n{content}\n")
    parts.append("[assistant]\n")
    return "\n".join(parts)


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def generate(
    messages: List[Dict[str, str]],
    min_len: Optional[int] = None,
    max_len: Optional[int] = None,
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    llama = _ensure_llama(model_override)
    settings = get_settings()
    min_c = int(min_len if min_len is not None else settings.min_len)
    max_c = int(max_len if max_len is not None else settings.max_len)
    if min_c > max_c:
        raise ValueError("min_len must be <= max_len")

    prompt = _build_prompt(messages)

    # First pass: ignore EOS entirely until reaching min chars
    max_tokens = max(16, max_c * 2 // 3)
    kwargs = dict(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        ignore_eos=True,
        stream=True,
    )
    if settings.top_k is not None:
        kwargs["top_k"] = settings.top_k
    if settings.min_p is not None:
        kwargs["min_p"] = settings.min_p
    if settings.repeat_penalty is not None:
        kwargs["repeat_penalty"] = settings.repeat_penalty

    stream = llama.create_completion(**kwargs)

    pieces: List[str] = []
    usage: Dict[str, Any] = {}
    for ev in stream:
        delta = ev.get("choices", [{}])[0].get("text", "")
        if delta:
            pieces.append(delta)
            if count_chars("".join(pieces)) >= min_c:
                break
        if not usage and "usage" in ev:
            usage = ev["usage"]

    text_first = "".join(pieces)

    # Optional second pass to encourage a natural stop
    second_pass = _bool_env("SECOND_PASS", False)
    sp_tokens = int(os.getenv("SECOND_PASS_TOKENS", "48") or 0)
    second_used = False
    if second_pass and sp_tokens > 0:
        second_used = True
        kwargs2 = dict(
            prompt=prompt + text_first,
            max_tokens=max(1, min(sp_tokens, 128)),
            temperature=settings.temperature,
            top_p=settings.top_p,
            ignore_eos=False,
            stream=True,
        )
        if settings.top_k is not None:
            kwargs2["top_k"] = settings.top_k
        if settings.min_p is not None:
            kwargs2["min_p"] = settings.min_p
        if settings.repeat_penalty is not None:
            kwargs2["repeat_penalty"] = settings.repeat_penalty

        stream2 = llama.create_completion(**kwargs2)
        tail: List[str] = []
        for ev in stream2:
            delta = ev.get("choices", [{}])[0].get("text", "")
            if delta:
                tail.append(delta)
            if not usage and "usage" in ev:
                usage = ev["usage"]
        text = text_first + "".join(tail)
    else:
        text = text_first

    trimmed = safe_trim(text, max_c)
    fixed = auto_close_pairs(trimmed)

    meta = {
        "model": getattr(llama, "model_path", None),
        "strategy": "ignore_eos",
        "second_pass_used": second_used,
        "min_len": min_c,
        "max_len": max_c,
        "generated_chars": count_chars(text),
        "returned_chars": count_chars(fixed),
        "usage": usage,
    }
    return {"text": fixed, "meta": meta}
