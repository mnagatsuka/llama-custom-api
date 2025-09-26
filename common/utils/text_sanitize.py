from __future__ import annotations

import re


SENTENCE_END = re.compile(r"[。．\.!?？！]\s*$")


def safe_trim(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    # Prefer cutting at a sentence boundary before max_len
    snippet = text[:max_len]
    # Find last sentence end in the snippet
    m = list(SENTENCE_END.finditer(snippet))
    if m:
        end = m[-1].end()
        return snippet[:end]
    # Fallback: cut at max_len and try not to cut inside whitespace
    # Trim trailing partial word/punctuation cleanly
    trimmed = snippet.rstrip()
    return trimmed


PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
    '"': '"',
    "'": "'",
    "（": "）",
    "「": "」",
    "『": "』",
}


def auto_close_pairs(text: str) -> str:
    stack: list[str] = []
    closers = set(PAIRS.values())
    for ch in text:
        if ch in PAIRS:
            stack.append(PAIRS[ch])
        elif ch in closers and stack and ch == stack[-1]:
            stack.pop()
    # Append missing closers in reverse order
    return text + "".join(reversed(stack))
