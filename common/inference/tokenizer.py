from __future__ import annotations

import unicodedata


SENTENCE_END_CHARS = set("。．.!?！？\n")


def count_chars(text: str) -> int:
    # Normalize to NFC to count composed characters consistently
    return len(unicodedata.normalize("NFC", text))


def is_sentence_end(token_str: str) -> bool:
    return any(ch in SENTENCE_END_CHARS for ch in token_str)
