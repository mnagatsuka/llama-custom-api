# EC2 Deployment — Pattern C (logits_processor)

Premise: Same base stack (Python 3.12, FastAPI, llama-cpp-python 0.3.x). This guide highlights only the differences for Pattern C.

## Environment (may change later)
- AMI/Instance: AWS Deep Learning Base GPU AMI (Ubuntu 24.04, us-west-2), g6.2xlarge
- Model file: GGUF stored under `/home/ubuntu/models/` (see model setup doc)

## Key Differences vs. Other Patterns
- App module: `src.c_logits_processor.app.main:app`
- Length window: controlled by `.env` (`MIN_LEN`, `MAX_LEN`); request fields are ignored.
- System prompt: controlled by `.env` (`SYSTEM_PROMPT`).
- Optional env var: `PUNCT_BIAS` (e.g., `0.3`).

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

Pattern C only (minimum set) — rsync just required sources:
```bash
# llama-custom-api/
rsync -av --delete \
  --exclude '.venv' --exclude '__pycache__' --exclude '.git' \
  ./common ./src/c_logits_processor pyproject.toml README.md .env.example \
  ubuntu@<EC2_PUBLIC_DNS>:/home/ubuntu/llama-custom-api/
```

Notes:
- Full project sync is simpler and keeps other patterns available. The Pattern C minimum sync copies only what this service path needs: `common/`, `src/c_logits_processor/`, `pyproject.toml`, `README.md`, and `.env.example`.
- Replace `<EC2_PUBLIC_DNS>` and `/path/to/key.pem` with your actual values.
- The full-project commands exclude local/dev files (e.g., `.mcp.json`, `.claude/`, `AGENTS.md`). Add more `--exclude` entries if your workspace has extra tooling folders.

## 2) Python 3.12 + Deps
- Create venv and install `-e .` (CPU or CUDA wheels per your instance). Same as generic.

## 3) Model + .env

Recommended model location on EC2: `/home/ubuntu/models/`. Avoid relying on Ollama cache paths; copy or download the GGUF to the models directory. For download and a llama.cpp sanity test, see `docs/deployment/model-setup-llama-cpp.md`.

If your GGUF lives on a separate volume (e.g., `/mnt/models`), either set `MODEL_PATH` to that absolute path or create a symlink: `ln -s /mnt/models /home/ubuntu/models`.

```
cd /home/ubuntu/llama-custom-api
cp .env.example .env
sed -i "s|MODEL_PATH=.*|MODEL_PATH=/home/ubuntu/models/Qwen3-30B-A3B-ERP-v0.1-Q8_0.gguf|" .env
sed -i "s|HOST=.*|HOST=0.0.0.0|" .env
sed -i "s|PORT=.*|PORT=8000|" .env

# Server-side character-length policy
sed -i "s|MIN_LEN=.*|MIN_LEN=120|" .env
sed -i "s|MAX_LEN=.*|MAX_LEN=160|" .env

# System prompt
echo "SYSTEM_PROMPT=You are a helpful assistant." >> .env

# Optional bias after min length
# echo "PUNCT_BIAS=0.3" >> .env
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
uv run -p 3.12 uvicorn src.c_logits_processor.app.main:app --host 0.0.0.0 --port 8000
```

Option B — activate venv, then run:
```bash
cd /home/ubuntu/llama-custom-api
source .venv/bin/activate
python -m uvicorn src.c_logits_processor.app.main:app --host 0.0.0.0 --port 8000
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
sudo tee /etc/systemd/system/llama-custom-api-c.service >/dev/null <<'EOF'
[Unit]
Description=llama-custom-api (Pattern C: logits_processor)
After=network-online.target
Wants=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/llama-custom-api
EnvironmentFile=/home/ubuntu/llama-custom-api/.env
ExecStart=/home/ubuntu/llama-custom-api/.venv/bin/uvicorn src.c_logits_processor.app.main:app --host ${HOST} --port ${PORT}
Restart=always
RestartSec=3
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable llama-custom-api-c && sudo systemctl start llama-custom-api-c
```

## 6) Verify
- Health: `curl -s http://<DNS>:8000/health`
- Chat: POST /chat with messages; note: server enforces system prompt and MIN_LEN/MAX_LEN from `.env`.

Example request (Pattern C):
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
- Request `system` messages are ignored; the server uses `.env` SYSTEM_PROMPT.
- The server enforces `MIN_LEN`/`MAX_LEN` from `.env`. Pattern C uses a logits processor for EOS control; you can tweak `PUNCT_BIAS`.

## 7) Notes
- Single-pass streaming with precise EOS control; set `PUNCT_BIAS` if you want slightly stronger sentence endings after the threshold.
- All other steps (CUDA wheels, Nginx, security groups) are identical to the generic guide.
