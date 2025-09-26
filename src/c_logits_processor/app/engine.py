from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from common.config import get_settings
from common.utils.logging import get_logger
from common.utils.text_sanitize import auto_close_pairs, safe_trim
from common.inference.tokenizer import count_chars
from .processors import MinCharLengthProcessor

logger = get_logger(__name__)

_llama_lock = threading.Lock()
_llama: Any = None
_eos_id: Optional[int] = None
_punct_ids: List[int] = []


def _ensure_llama(model_path: Optional[str] = None) -> Any:
    global _llama, _eos_id, _punct_ids
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
            # Determine EOS token id
            eos_id = None
            try:
                eos_id = _llama.token_eos()  # type: ignore[attr-defined]
            except Exception:
                try:
                    # Fallback: many tokenizers use </s>
                    eos_id = _llama.tokenize("</s>", add_bos=False, special=True)[0]
                except Exception:
                    eos_id = None
            _eos_id = eos_id
            # Precompute punctuation token ids for biasing
            _punct_ids = []
            for ch in ["。", "．", ".", "!", "?", "！", "？", "\n"]:
                try:
                    toks = _llama.tokenize(ch, add_bos=False, special=False)
                    if toks:
                        _punct_ids.append(toks[0])
                except Exception:
                    pass
    return _llama


def _build_prompt(messages: List[Dict[str, str]]) -> str:
    # Simple chat-style prompt concatenation
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

    # Configure logits processor for EOS suppression and punctuation bias
    processor = MinCharLengthProcessor(
        eos_token_id=_eos_id,
        min_len=min_c,
        punctuation_token_ids=_punct_ids,
        punctuation_bias=0.3,
    )

    # Use create_completion with logits_processor support
    # Incrementally update processor char count based on generated text length
    max_tokens = max(16, max_c * 2 // 3)
    kwargs = dict(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        logits_processor=[processor],  # type: ignore[arg-type]
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
    usage = {}
    for ev in stream:
        delta = ev.get("choices", [{}])[0].get("text", "")
        if delta:
            pieces.append(delta)
            processor.update_char_count("".join(pieces))
        if not usage and "usage" in ev:
            usage = ev["usage"]
    text = "".join(pieces)
    # Enforce max length with safe trim and auto-close
    trimmed = safe_trim(text, max_c)
    fixed = auto_close_pairs(trimmed)

    meta = {
        "model": getattr(llama, "model_path", None),
        "min_len": min_c,
        "max_len": max_c,
        "generated_chars": count_chars(text),
        "returned_chars": count_chars(fixed),
        "eos_suppressed": count_chars(text) < min_c,
        "usage": usage,
    }
    return {"text": fixed, "meta": meta}
