# Pattern C (Custom LogitsProcessor) — Implementation Guideline

This guideline describes how to implement the “custom logits processor” approach for EOS suppression, aligned with docs/_dev_guidelines/patterns-separation.md.

## Scope and Intent

- Goal: Enforce a minimum character length by suppressing EOS during generation, then allow normal stopping with a small bias toward sentence endings.
- Trade-off: Highest implementation complexity, but enables single-pass generation, streaming, and precise control.
- API: Same request/response schema as the base project.

## Location and Layout

- Code under `src/c_logits_processor/` (planned), or current single-app `app/`:
  - `app/main.py`: FastAPI entrypoint (`/chat`, `/health`).
  - `app/engine.py`: Generation using a logits processor.
  - `app/processors.py`: `MinCharLengthProcessor` (EOS suppression/biasing).
  - `tests/`: Unit/E2E tests for Pattern C.
  - `.env.example`: Pattern-specific defaults (optional).
- Shared code via `common/` (planned):
  - `common/config.py`, `common/models.py`, `common/utils/*`, `common/inference/tokenizer.py`.

## Engine Behavior

Approach C — Single-pass with processor:
- Initialize `MinCharLengthProcessor(eos_token_id, min_len, punctuation_token_ids, punctuation_bias)`.
- Call `Llama.create_completion(..., logits_processor=[processor], stream=True)`.
- As chunks arrive, update the processor’s char counter so EOS remains suppressed until `min_len`.
- After threshold, EOS suppression is lifted; apply a small positive bias to sentence-ending tokens for cleaner stops.
- Post-process (`safe_trim`, `auto_close_pairs`) and return.

Notes:
- Works well with SSE streaming (no second request).
- Ensure thread-safety for the shared `Llama` instance (mutex or pool).

### Pseudocode
```
from common.config import get_settings
from common.utils.text_sanitize import safe_trim, auto_close_pairs
from common.inference.tokenizer import count_chars
from .processors import MinCharLengthProcessor


def generate(messages, min_len=None, max_len=None, model_override=None) -> dict:
    # 1) resolve settings + prompt
    # 2) init processor with eos_id and punctuation ids
    # 3) stream create_completion(logits_processor=[processor])
    # 4) update processor char count on each chunk
    # 5) finalize with safe_trim/auto_close_pairs; return meta
```

## Configuration

Pattern-specific keys (optional):
- `PUNCT_BIAS`: float, e.g., `0.3` (bias for sentence-ending tokens after release).
- Shared: `MIN_LEN`, `MAX_LEN`, `N_THREADS`, `CTX_SIZE`, `MODEL_PATH`.

Validation:
- `min_len <= max_len` or 400 error.
- Clamp `PUNCT_BIAS` to a small positive range if desired (e.g., 0.0–1.0).

## API Contract

- Endpoint: `POST /chat` (unchanged).
- Request: `ChatRequest` (messages, optional min_len/max_len, optional model).
- Response: `ChatResponse` with `text`, `meta`.
- Meta suggestions:
  - `strategy: "logits_processor"`
  - `punct_bias: float`
  - `generated_chars`, `returned_chars`

## Testing

Unit tests (src/c_logits_processor/tests/):
- Processor: EOS suppressed while chars < `min_len`; released after.
- Punctuation bias: logits increased for sentence-ending tokens post-threshold.
- Max length trimming and auto-close behavior verified.

E2E (optional):
- Run app and POST `/chat` within a narrow window (e.g., 120–140) and validate text length and meta.

## Runbook

- Local dev:
  - Single-app (current): `uv run uvicorn app.main:app --reload`
  - Multi-pattern: `APP_MODULE=src.c_logits_processor.app.main:app uv run uvicorn $APP_MODULE --reload`
- EC2 (systemd) example:
  - `ExecStart=/home/ubuntu/llama-custom-api/.venv/bin/uvicorn src.c_logits_processor.app.main:app --host ${HOST} --port ${PORT}`
- Env:
  - Copy `.env.example` as needed; set `MIN_LEN`, `MAX_LEN`, and `PUNCT_BIAS`.

## Migration Checklist

1) Ensure `common/` exists and shared modules are moved (config, models, utils, tokenizer).
2) Place current logits-processor implementation under `src/c_logits_processor/`.
3) Keep `generate(messages, min_len, max_len, model_override)` interface stable.
4) Verify `APP_MODULE` switching works (A/B/C).
5) Update docs/examples to point to Pattern C module path.

## Performance and Limits

- Single-pass streaming with precise control; minimal latency overhead.
- Small positive punctuation bias helps clean stops; keep conservative to avoid artifacts.
- GPU/CPU parity; CUDA wheels recommended on g6 instances for throughput.

## Failure Modes & Mitigations

- Underflow (early stop): ensure processor updates correctly and EOS is suppressed pre-threshold.
- Overflow (rambling): enforce `MAX_LEN` via `safe_trim` and keep `max_tokens` reasonable.
- Concurrency: protect shared `Llama` with a lock; or use a small instance pool.
