# Model Setup with llama.cpp (GGUF)

This guide explains how to set up a local GGUF model with llama.cpp and connect it to this FastAPI-based API. It uses Qwen3-30B GGUF as a concrete example, but the steps apply to any GGUF model (Llama, Mistral, Qwen, etc.).

## 1) Requirements

- OS: Linux or macOS (Windows works via WSL2 or native build).
- Hardware: CPU works; NVIDIA/Apple GPU recommended for large models.
- Tools: `git`, `cmake`, compiler toolchain, Python 3.10+ (3.12 preferred).
- Optional: CUDA (NVIDIA) or Metal (Apple) toolchain for GPU acceleration.

## 2) Install Dependencies

Linux (Ubuntu example):
```bash
sudo apt update
sudo apt install -y build-essential git wget cmake python3-venv
```

macOS:
```bash
xcode-select --install
brew install cmake git wget python
```

Windows:
- Use WSL2 Ubuntu and follow Linux steps, or build natively with MSVC + CMake.

## 3) Build llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
# Choose ONE of the following build lines:
cmake .. && make -j                 # CPU
cmake .. -DLLAMA_CUBLAS=ON && make -j  # NVIDIA CUDA
cmake .. -DLLAMA_METAL=ON && make -j   # Apple Metal
```

Binaries (e.g., `llama-cli`, `llama-server`) are placed in `./build/bin/`.

## 4) Obtain a GGUF Model

Recommended location on EC2: `/home/ubuntu/models/`

Using Qwen3-30B (Q8_0) as example:

```bash
mkdir -p /home/ubuntu/models
cd /home/ubuntu/models
wget -O Qwen3-30B-A3B-ERP-v0.1-Q8_0.gguf \
  "https://huggingface.co/Aratako/Qwen3-30B-A3B-ERP-v0.1-GGUF/resolve/main/Qwen3-30B-A3B-ERP-v0.1-Q8_0.gguf"
ls -lh /home/ubuntu/models  # verify size
```

Notes:
- Prefer a stable absolute path like `/home/ubuntu/models/` for predictable deployments and permissions.
- If using a mounted volume (e.g., `/mnt/models`), either point `MODEL_PATH` there directly or create a symlink: `ln -s /mnt/models /home/ubuntu/models`.
- Quantization tradeoff: Q8_0 is larger/faster (more RAM/VRAM). Use `Q6_K`/`Q5_K`/`Q4_K` for lower memory footprints.
- Any GGUF model works the same way; just change the filename and path.

## 5) Quick Sanity Test (llama.cpp CLI)

```bash
~/llama.cpp/build/bin/llama-cli \
  -m ~/models/Qwen3-30B-A3B-ERP-v0.1.Q8_0.gguf \
  -p "Hello, how are you today?" \
  -n 64
```

If you see a coherent response without out-of-memory (OOM) errors, the model file is healthy.

## 6) Performance & Tuning

- Model size vs. speed: Q8_0 is faster but needs more memory. `Q6_K`/`Q5_K`/`Q4_K` reduce memory at the cost of speed/quality.
- Threads: Increase `N_THREADS` (CPU) up to the number of physical cores.
- Context window: Increase `CTX_SIZE` to 8192 or higher if memory allows.
- GPU: Use CUDA (NVIDIA) or Metal (Apple) builds to offload to GPU.

## 7) Troubleshooting

- OOM / slow performance:
  - Use a smaller quantization or model.
  - Ensure sufficient swap (CPU) and that GPU VRAM matches model size.
- Model path errors:
  - Use an absolute path and verify file permissions/ownership.
- CUDA issues:
  - Match the wheel to your CUDA version (e.g., cu124). Rebuild llama.cpp with `-DLLAMA_CUBLAS=ON` if necessary.
- Truncated output / early stops:
  - Adjust `MIN_LEN` / `MAX_LEN` and sampling params (as supported by your app).
