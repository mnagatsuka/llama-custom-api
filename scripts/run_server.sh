#!/usr/bin/env bash
set -euo pipefail
if [ -f .env ]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs -I{} echo {})
fi
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
APP_MODULE_DEFAULT="src.c_logits_processor.app.main:app"
uv run uvicorn "${APP_MODULE:-$APP_MODULE_DEFAULT}" --host "$HOST" --port "$PORT"
