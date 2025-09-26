from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # dotenv is optional; ignore if unavailable
    pass


def _getenv_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default


def _getenv_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default


def _getenv_optional_int(key: str) -> int | None:
    val = os.getenv(key)
    if val is None or val == "":
        return None
    try:
        return int(val)
    except Exception:
        return None


def _getenv_optional_float(key: str) -> float | None:
    val = os.getenv(key)
    if val is None or val == "":
        return None
    try:
        return float(val)
    except Exception:
        return None


def _resolve_system_prompt() -> str:
    path = os.getenv("SYSTEM_PROMPT_FILE")
    p = Path(path).expanduser()
    text = p.read_text(encoding="utf-8")
    return text.rstrip("\n")


@dataclass(frozen=True)
class Settings:
    model_path: str
    n_threads: int
    ctx_size: int
    min_len: int
    max_len: int
    host: str
    port: int
    system_prompt: str
    temperature: float
    top_p: float
    top_k: int | None
    min_p: float | None
    repeat_penalty: float | None


def get_settings() -> Settings:
    ctx_env = os.getenv("CTX_SIZE") or "4096"
    try:
        ctx_val = int(ctx_env)
    except Exception:
        ctx_val = 4096

    return Settings(
        model_path=os.getenv("MODEL_PATH", "model.gguf"),
        n_threads=_getenv_int("N_THREADS", 4),
        ctx_size=ctx_val,
        min_len=_getenv_int("MIN_LEN", 120),
        max_len=_getenv_int("MAX_LEN", 240),
        host=os.getenv("HOST", "127.0.0.1"),
        port=_getenv_int("PORT", 8000),
        system_prompt=_resolve_system_prompt(),
        temperature=_getenv_float("TEMPERATURE", 0.7),
        top_p=_getenv_float("TOP_P", 0.95),
        top_k=_getenv_optional_int("TOP_K"),
        min_p=_getenv_optional_float("MIN_P"),
        repeat_penalty=_getenv_optional_float("REPEAT_PENALTY"),
    )
