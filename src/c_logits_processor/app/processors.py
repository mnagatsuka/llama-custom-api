from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Iterable, Optional

try:
    from llama_cpp import LogitsProcessor  # type: ignore
except Exception:  # pragma: no cover - fallback stub for type checking
    class LogitsProcessor:  # type: ignore
        def __call__(self, input_ids, scores):
            return scores


class MinCharLengthProcessor(LogitsProcessor):
    """
    Suppress EOS until a minimum character length is reached, then
    release suppression and optionally bias sentence-ending punctuation.
    """

    def __init__(
        self,
        eos_token_id: Optional[int],
        min_len: int,
        punctuation_token_ids: Optional[Iterable[int]] = None,
        punctuation_bias: float = 0.5,
    ) -> None:
        self.eos_token_id = eos_token_id
        self.min_len = max(0, int(min_len))
        self._chars = 0
        self._released = self.min_len == 0
        self.punct_ids = set(punctuation_token_ids or [])
        self.punct_bias = float(punctuation_bias)

    def update_char_count(self, new_text: str) -> None:
        # Called externally after decoding to keep char count in sync
        self._chars = len(new_text)
        if not self._released and self._chars >= self.min_len:
            self._released = True

    def __call__(self, input_ids, logits):  # noqa: N802 - API contract
        # input_ids: token sequence array; logits: vocabulary logits array
        import numpy as np
        
        # Work with logits (the vocabulary scores, not input_ids)
        if not self._released:
            if self.eos_token_id is not None and 0 <= self.eos_token_id < len(logits):
                # Suppress EOS strongly
                logits[self.eos_token_id] = float("-inf")
        else:
            # Add small positive bias to punctuation tokens to encourage clean endings
            if self.punct_ids:
                for tid in self.punct_ids:
                    if 0 <= tid < len(logits):
                        logits[tid] = float(logits[tid]) + self.punct_bias
        
        return logits
