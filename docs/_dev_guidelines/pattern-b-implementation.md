# Pattern B (logit_bias) — Implementation Guideline

This guideline describes how to implement the “logit_bias” approach for EOS suppression, based on docs/_dev_guideline/patterns-separation.md.

## Scope and Intent

- Goal: Lower the probability of EOS using a negative logit bias until a minimum character length is likely achieved.
- Trade-off: Not an absolute guarantee; aims for naturalness with strong guidance.
- API: Same request/response schema as the base project.

## Location and Layout

- Code under `src/b_logit_bias/`:
  - `app/main.py`: FastAPI entrypoint (`/chat`, `/health`).
  - `app/engine.py`: Generation using `logit_bias` for EOS.
  - `tests/`: Unit/E2E tests for Pattern B.
  - `.env.example`: Pattern-specific defaults.
- Shared code via `common/`:
  - `common/config.py`, `common/models.py`, `common/utils/*`, `common/inference/tokenizer.py`.

## Engine Behavior

Approach B.1 — Single-pass (simplest):
- Prepare a `logit_bias` mapping: `{ eos_token_id: EOS_BIAS }` with a strong negative number (e.g., -10.0).
- Call `Llama.create_completion(..., logit_bias={...}, stream=True)`.
- Accumulate text and stop when token budget or length constraints are met.
- Post-process (`safe_trim`, `auto_close_pairs`).

Approach B.2 — Two-pass (bias removal after threshold):
- Pass 1: same as B.1 (apply `logit_bias`), stream until `min_len` or token budget reached.
- Pass 2: short follow-up call without `logit_bias` (e.g., 24–48 tokens) using the tail of Pass 1 as context to encourage a clean natural stop.
- Concatenate and post-process to `max_len`.

Notes:
- `logit_bias` is static for the duration of a call in llama-cpp-python; removing it at the threshold requires a second call (B.2).
- Keep thread-safety for the shared `Llama` instance (mutex or pool), as in Pattern C.

### Pseudocode
```
from common.config import get_settings
from common.utils.text_sanitize import safe_trim, auto_close_pairs
from common.inference.tokenizer import count_chars


def generate(messages, min_len=None, max_len=None, model_override=None) -> dict:
    # 1) resolve settings + prompt
    # 2) pass1 = create_completion(stream=True, logit_bias={eos: EOS_BIAS})
    # 3) accumulate; if chars >= min_len and SECOND_PASS: run pass2 without bias (small budget)
    # 4) final = safe_trim(auto_close_pairs(concat(pass1, pass2)), max_len)
    # 5) return {"text": final, "meta": {...}}
```

## Configuration

Pattern-specific keys (`src/b_logit_bias/.env.example`):
- `EOS_BIAS`: float (default: `-10.0`).
- `SECOND_PASS`: `true|false` (default: `false`).
- `SECOND_PASS_TOKENS`: int, e.g., `32` (cap at ~128).
- Shared: `MIN_LEN`, `MAX_LEN`, `N_THREADS`, `CTX_SIZE`, `MODEL_PATH`.

Validation:
- `min_len <= max_len` or 400 error.
- Clamp `EOS_BIAS` to a safe range (e.g., -4 to -20) if needed.

## API Contract

- Endpoint: `POST /chat` (unchanged).
- Request: `ChatRequest` (messages, optional min_len/max_len, optional model).
- Response: `ChatResponse` with `text`, `meta`.
- Meta suggestions:
  - `strategy: "logit_bias"`
  - `eos_bias: float`
  - `second_pass_used: bool`
  - `generated_chars`, `returned_chars`

## Testing

Unit tests (src/b_logit_bias/tests/):
- Single-pass: typical outputs meet or exceed `min_len` under strong negative bias.
- Two-pass: improved sentence termination frequency when `SECOND_PASS=true`.
- Max length trimming and auto-close behavior verified.
- Error handling for `min_len > max_len`.
- Mock `Llama` to produce deterministic chunks for assertions.

E2E (optional):
- Run app, POST `/chat` with and without second pass; assert length windows and meta flags.

## Runbook

- Local dev:
  - `APP_MODULE=src.b_logit_bias.app.main:app uv run uvicorn $APP_MODULE --reload`
- EC2 (systemd) example:
  - `ExecStart=/home/ubuntu/llama-custom-api/.venv/bin/uvicorn src.b_logit_bias.app.main:app --host ${HOST} --port ${PORT}`
- Env:
  - Copy `src/b_logit_bias/.env.example` → `.env`; set `EOS_BIAS`, `SECOND_PASS`, `MIN_LEN`, `MAX_LEN`.

## Migration Checklist

1) Ensure `common/` package exists and shared modules moved.
2) Create `src/b_logit_bias/` with `app/main.py`, `app/engine.py`, `.env.example`, and tests.
3) Import from `common.*` across the pattern.
4) Verify `APP_MODULE` switching works (A/B/C).
5) Document default `EOS_BIAS` and second-pass behavior in README/docs.

## Performance and Limits

- Strong negative bias reduces premature EOS while maintaining natural token distribution.
- Overly strong bias may cause run-on sentences; mitigate with smaller `temperature`, bounded `SECOND_PASS_TOKENS`, or a slightly weaker bias.
- GPU/CPU parity; CUDA wheels recommended on g6 instances for throughput.

## Failure Modes & Mitigations

- Still short outputs: reduce temperature or make `EOS_BIAS` more negative (e.g., -12 to -16) or enable second pass.
- Unnatural endings: enable second pass with a small budget to allow a clean stop.
- Concurrency: protect shared `Llama` with a lock; or use a small instance pool.
