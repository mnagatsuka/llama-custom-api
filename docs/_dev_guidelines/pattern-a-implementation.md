# Pattern A (ignore_eos) — Implementation Guideline [Deprecated]

Important: Pattern A depends on an `ignore_eos` parameter that is not supported by the current `llama-cpp-python` (0.3.x) bindings. As such, this pattern is deprecated and not recommended for active use. Prefer Pattern B (logit_bias) or Pattern C (logits_processor), which achieve similar outcomes without requiring unsupported parameters. The content below is retained for historical and design reference only.

This guide defines how to implement and operate the “ignore_eos” pattern alongside other variants. It builds on docs/_dev_guideline/patterns-separation.md.

## Scope and Intent

- Goal: Guarantee a minimum character length by ignoring EOS until the threshold is reached.
- Trade-off: May produce verbose endings; optionally run a short second pass to encourage a clean stop.
- API: Same request/response schema as base project.

## Location and Layout

- Code lives under `src/a_ignore_eos/`:
  - `app/main.py`: FastAPI entrypoint (mounts `/chat`, `/health`).
  - `app/engine.py`: Generation with `ignore_eos` strategy (+ optional second pass).
  - `tests/`: Unit/E2E tests for pattern A.
  - `.env.example`: Pattern-specific defaults.
- Shared code consumed from `common/`:
  - `common/config.py`, `common/models.py`, `common/utils/*`, `common/inference/tokenizer.py`.

## Engine Behavior

Status: Not implementable as-specified with `llama-cpp-python` 0.3.x because `Llama.create_completion` does not accept `ignore_eos`. If you must approximate this behavior, use EOS logit bias as in Pattern B.

Strategy A.1 — Single-Pass (hard min) [not supported in current bindings]:
- Intended: call `Llama.create_completion(..., ignore_eos=True, stream=True)`.
- Keep generating until char count >= `min_len`.
- Stop early if token budget exhausted; apply post-processing (`safe_trim`, `auto_close_pairs`).

Strategy A.2 — Two-Ppass (clean end) [historical reference]:
- Intended: Pass 1 until `min_len` with EOS ignored; Pass 2 without ignore to encourage a natural stop.
- Concatenate results, then `safe_trim` to `max_len` and `auto_close_pairs`.

Pseudo-interface (historical):
```
from common.config import get_settings
from common.utils.text_sanitize import safe_trim, auto_close_pairs
from common.inference.tokenizer import count_chars


def generate(messages, min_len=None, max_len=None, model_override=None) -> dict:
    # 1) resolve settings and build prompt (reuse common helpers)
    # 2) create_completion(ignore_eos=True, stream=True)
    # 3) accumulate text; stop when count_chars(text) >= min_len
    # 4) optional second pass if enabled and budget > 0
    # 5) trim/close; return {"text": final_text, "meta": {...}}
```

Important notes:
- In current bindings, prefer Pattern B’s EOS logit bias to emulate “ignore until min length”.
- For streaming UX, emit deltas as they arrive; only enforce second pass if you buffer.
- Ensure thread-safety around the shared `Llama` instance (mutex or pool), as in Pattern C.

## Configuration

Environment keys (example defaults for `.env.example` under pattern A):
- `MIN_LEN` / `MAX_LEN`: default length window (fallback if request omits).
- `SECOND_PASS`: `true|false` (default: `false`).
- `SECOND_PASS_TOKENS`: integer token budget (default: `48`).
- `N_THREADS`, `CTX_SIZE`, `MODEL_PATH`: as in base project.

Validation:
- Reject `min_len > max_len` (400).
- Cap `SECOND_PASS_TOKENS` to a safe upper bound (e.g., 128) to avoid long tails.

## API Contract

- Endpoint: `POST /chat`
- Request body: same `ChatRequest` (messages, optional min_len/max_len, optional model).
- Response: same `ChatResponse` with `text` and `meta`.
- Meta suggestions:
  - `strategy: "ignore_eos"`
  - `second_pass_used: bool`
  - `generated_chars` / `returned_chars`

## Testing

Unit tests (src/a_ignore_eos/tests/):
- Char threshold: ensure output reaches `min_len` before termination in single-pass mode.
- Second pass: when enabled, ensure a reasonable stop within the token budget and improved sentence ending frequency.
- Max length: `safe_trim` respects `max_len` and closing pairs are appended.
- Error: `min_len > max_len` returns 400.
- Determinism: mock `Llama` to return predictable chunks.

E2E (optional):
- Spin up FastAPI app; POST /chat with `min_len` near `max_len` and validate lengths and meta flags.

## Runbook

- Selecting the app:
  - `APP_MODULE=src.a_ignore_eos.app.main:app uv run uvicorn $APP_MODULE --reload`
- EC2 (systemd) example:
  - `ExecStart=/home/ubuntu/llama-custom-api/.venv/bin/uvicorn src.a_ignore_eos.app.main:app --host ${HOST} --port ${PORT}`
- Env:
  - Copy `src/a_ignore_eos/.env.example` → `.env` and adjust as needed.

## Migration Checklist (from current tree)

1) Create `common/` and move shared modules from `app/` to `common/`.
2) Move current Pattern C implementation under `src/c_logits_processor/`.
3) Add Pattern A folder with `app/main.py`, `app/engine.py`, `.env.example`, and tests.
4) Update imports to `common.*`.
5) Verify `APP_MODULE` switching works for A/B/C.
6) Update docs and deployment scripts (if you add wrappers) accordingly.

## Performance and Limits

- Pattern A avoids softmax-time logits edits; simplest path but may over-generate without careful `max_tokens` tuning.
- Consider temperature/top_p conservative defaults; pair with `SECOND_PASS` for cleaner stops.
- GPU vs CPU behavior is identical; CUDA wheels recommended on g6 instances.

## Failure Modes & Mitigations

- Runaway generation: enforce `max_tokens` and `MAX_LEN` trimming.
- Unnatural endings: enable `SECOND_PASS` with a small budget.
- Concurrency: guard shared `Llama` instance with a lock; or instantiate per request at cost of latency.
