# EC2 Deployment — Pattern A (ignore_eos) [Deprecated]

Important: Pattern A relies on an `ignore_eos` option that is not supported by the current `llama-cpp-python` (0.3.x) bindings. As a result, this deployment path is deprecated and not recommended. Use Pattern B (logit_bias) or Pattern C (logits_processor), which achieve similar goals without requiring unsupported parameters.

Recommended alternatives:
- Pattern B: `docs/deployment/ec2-b.md` (suppresses EOS via `logit_bias` during the first pass)
- Pattern C: `docs/deployment/ec2-c.md` (uses a logits processor approach)

Premise: Same base stack (Python 3.12, FastAPI, llama-cpp-python 0.3.x). This guide remains for reference only and may not work end-to-end due to the above limitation.

## Environment (may change later)
- AMI/Instance: AWS Deep Learning Base GPU AMI (Ubuntu 24.04, us-west-2), g6.2xlarge
- Model file: GGUF stored under `/home/ubuntu/models/` (see model setup doc)

## Key Differences vs. Other Patterns
- App module: `src.a_ignore_eos.app.main:app`
- Length window: controlled by `.env` (`MIN_LEN`, `MAX_LEN`); request fields are ignored.
- System prompt: controlled by `.env` (`SYSTEM_PROMPT`).
- Extra env vars: `SECOND_PASS`, `SECOND_PASS_TOKENS` (optional).

## 1) Upload Code

Option A — rsync (recommended, full project):
```bash
# llama-custom-api/
rsync -av --delete \
  --exclude '.venv' --exclude '__pycache__' --exclude '.git' \
  --exclude '.mcp.json' --exclude '.claude' --exclude '.serena' --exclude '.cache' \
  --exclude 'temp' --exclude '.tools' \
  --exclude 'AGENTS.md' --exclude 'CLAUDE.md' --exclude 'GEMINI.md' \
  ./ ubuntu@<EC2_PUBLIC_DNS>:/home/ubuntu/llama-custom-api/
```

Option B — scp (archive, full project):
```bash
# llama-custom-api/
( cd .. && tar \
    --exclude='llama-custom-api/.venv' \
    --exclude='llama-custom-api/__pycache__' \
    --exclude='llama-custom-api/.git' \
    --exclude='llama-custom-api/.mcp.json' \
    --exclude='llama-custom-api/.claude' \
    --exclude='llama-custom-api/.serena' \
    --exclude='llama-custom-api/.cache' \
    --exclude='llama-custom-api/temp' \
    --exclude='llama-custom-api/.tools' \
    --exclude='llama-custom-api/AGENTS.md' \
    --exclude='llama-custom-api/CLAUDE.md' \
    --exclude='llama-custom-api/GEMINI.md' \
    -czf llama-custom-api.tgz llama-custom-api )
scp -i /path/to/key.pem ../llama-custom-api.tgz ubuntu@<EC2_PUBLIC_DNS>:/home/ubuntu/
ssh -i /path/to/key.pem ubuntu@<EC2_PUBLIC_DNS> 'rm -rf /home/ubuntu/llama-custom-api && mkdir -p /home/ubuntu/llama-custom-api && tar xzf /home/ubuntu/llama-custom-api.tgz -C /home/ubuntu && rm /home/ubuntu/llama-custom-api.tgz'
```

Pattern A only (minimum set) — rsync just required sources:
```bash
# llama-custom-api/
rsync -av --delete \
  --exclude '.venv' --exclude '__pycache__' --exclude '.git' \
  ./common ./src/a_ignore_eos pyproject.toml README.md .env.example \
  ubuntu@<EC2_PUBLIC_DNS>:/home/ubuntu/llama-custom-api/
```

Notes:
- Full project sync is simpler and keeps other patterns available. The Pattern A minimum sync copies only what this service path needs: `common/`, `src/a_ignore_eos/`, `pyproject.toml`, `README.md`, and `.env.example`.
- Replace `<EC2_PUBLIC_DNS>` and `/path/to/key.pem` with your actual values.
- The full-project commands exclude local/dev files (e.g., `.mcp.json`, `.claude/`, `AGENTS.md`). Add more `--exclude` entries if your workspace has extra tooling folders.

## 2) Python 3.12 + Deps (uv)

Use uv for Python management, virtualenv, and installs. For full options (CPU/CUDA), see `docs/deployment/ec2.md`.

```bash
# Install uv and Python 3.12
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12

# Create and activate a project venv
cd /home/ubuntu/llama-custom-api
uv venv -p 3.12
source .venv/bin/activate
python -V

# Minimal install (CPU-only)
uv pip install -e .

# Verify uvicorn is installed in the venv
python -c "import uvicorn, fastapi; print('uvicorn', uvicorn.__version__, 'fastapi', fastapi.__version__)"

# CUDA wheel (optional, NVIDIA L4 etc.) — pick your CUDA tag (e.g., cu124)
# uv pip install --upgrade --extra-index-url \
#   https://abetlen.github.io/llama-cpp-python/whl/cu124 \
#   "llama-cpp-python>=0.3,<0.4"
# uv pip install -e .
```

Troubleshooting:
- If activation fails with `-bash: .venv/bin/activate: No such file or directory`, the venv wasn’t created. Re-run `uv venv -p 3.12` from `/home/ubuntu/llama-custom-api`, then `source .venv/bin/activate`.

## 3) Model + .env

Recommended model location on EC2: `/home/ubuntu/models/`. Avoid relying on Ollama cache paths; copy or download the GGUF to the models directory. For download and a llama.cpp sanity test, see `docs/deployment/model-setup-llama-cpp.md`.

If your GGUF lives on a separate volume (e.g., `/mnt/models`), either set `MODEL_PATH` to that absolute path or create a symlink: `ln -s /mnt/models /home/ubuntu/models`.

```
cd /home/ubuntu/llama-custom-api
cp .env.example .env

# Point to the absolute path of your GGUF file (example filename shown)
sed -i "s|MODEL_PATH=.*|MODEL_PATH=/home/ubuntu/models/Qwen3-30B-A3B-ERP-v0.1-Q8_0.gguf|" .env
sed -i "s|HOST=.*|HOST=0.0.0.0|" .env
sed -i "s|PORT=.*|PORT=8000|" .env

# Pattern A specific knobs (server-side control)
# Character-length policy (used by the server, request fields are ignored)
sed -i "s|MIN_LEN=.*|MIN_LEN=120|" .env
sed -i "s|MAX_LEN=.*|MAX_LEN=160|" .env

# Optional second pass parameters
echo "SECOND_PASS=false" >> .env
echo "SECOND_PASS_TOKENS=48" >> .env

# Optional: override the system prompt used by the server
echo "SYSTEM_PROMPT=You are a helpful assistant." >> .env
```

Alternative (recommended): manage the system prompt via file
```bash
mkdir -p /home/ubuntu/llama-custom-api/prompts
cat >/home/ubuntu/llama-custom-api/prompts/system_prompt.md <<'EOF'
You are a helpful assistant.
Follow these rules:
- Be concise
- Prefer Japanese in responses
EOF
echo "SYSTEM_PROMPT_FILE=/home/ubuntu/llama-custom-api/prompts/system_prompt.md" >> .env
```

Notes:
- Use an absolute `MODEL_PATH` (e.g., `/home/ubuntu/models/<model>.gguf`) to avoid path resolution issues.
- You can tune sampling via `.env` keys: `TEMPERATURE` (default 0.7) and `TOP_P` (default 0.95).

## 4) Foreground Run

Option A — use uv to run (no manual activation needed):
```bash
cd /home/ubuntu/llama-custom-api
uv run -p 3.12 uvicorn src.a_ignore_eos.app.main:app --host 0.0.0.0 --port 8000
```

Option B — activate venv, then run:
```bash
cd /home/ubuntu/llama-custom-api
source .venv/bin/activate
python -m uvicorn src.a_ignore_eos.app.main:app --host 0.0.0.0 --port 8000
```

If `uvicorn` is still not found, ensure dependencies are installed:
```bash
cd /home/ubuntu/llama-custom-api
source .venv/bin/activate
uv pip install -e .
# or explicitly:
uv pip install 'uvicorn[standard]>=0.30,<0.31'
```

## 5) Systemd Unit

```
sudo tee /etc/systemd/system/llama-custom-api-a.service >/dev/null <<'EOF'
[Unit]
Description=llama-custom-api (Pattern A: ignore_eos)
After=network-online.target
Wants=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/llama-custom-api
EnvironmentFile=/home/ubuntu/llama-custom-api/.env
ExecStart=/home/ubuntu/llama-custom-api/.venv/bin/uvicorn src.a_ignore_eos.app.main:app --host ${HOST} --port ${PORT}
Restart=always
RestartSec=3
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable llama-custom-api-a && sudo systemctl start llama-custom-api-a
```

## 6) Verify
- Health: `curl -s http://<DNS>:8000/health`
- Chat: POST /chat with messages; note: server enforces system prompt, MIN_LEN/MAX_LEN from `.env`.

Example request (Pattern A):
```bash
curl -s -X POST http://<DNS>:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
        "messages": [
          {"role":"user","content":"自己紹介を120〜160文字でお願いします。"}
        ]
      }'
```

Local request (from the same EC2 instance):
```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
        "messages": [
          {"role":"user","content":"Introduce yourself in 120-160 Japanese characters."}
        ]
      }'
```
Notes:
- `messages[0].role` should be `user`. Any `system` message in the request is ignored; the server uses `.env` SYSTEM_PROMPT.
- The server also enforces `MIN_LEN`/`MAX_LEN` from `.env`; request `min_len`/`max_len` are ignored in Pattern A.

## 7) Notes
- Optional second pass can improve natural stops. Control with `SECOND_PASS` and `SECOND_PASS_TOKENS`.
- All other steps (CUDA wheels, Nginx, security groups) are identical to the generic guide.
