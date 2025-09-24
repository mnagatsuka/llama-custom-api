# llama-custom-api

- This project is an API that adds a **character count control feature** to a chat application using llama.  
- It addresses the issue of **unstable output length** in OSS LLMs using llama or Ollama, where responses may be too short (underflow) or too long (overflow).  
- Since Ollama’s Structured Output and the `num_predict` parameter proved insufficient for strict length control, we implemented EOS token control within the generation loop.  

## Implementation Policy

### Directly use `llama.cpp / llama-cpp-python` and implement a custom MinCharLengthProcessor

- **Remove dependency on Ollama**  
  - Switch to FastAPI → llama.cpp/llama-cpp-python direct calls.  
  - Avoid internal modifications to Ollama itself, reducing maintenance costs when updating.  

- **Implement MinCharLengthProcessor**  
  - During the inference loop, if `generated length < minimum length`, replace the EOS token logit with `-∞`.  
  - Once the minimum length is reached, remove the EOS suppression and allow normal termination.  
  - To improve sentence quality, add a small positive bias to punctuation or sentence-ending tokens after suppression is lifted.  

- **Maximum length control**  
  - Use `num_predict` to ensure a safe margin for generation.  
  - After generation, count characters and trim safely if the output exceeds the specified maximum length, while keeping sentence integrity.  
  - Fix unclosed parentheses or quotes with post-processing.  

- **API design**  
  - Only one endpoint: `POST /chat`.  
  - Input: JSON (model name + messages).  
  - Output: JSON (`{"text": "...", "meta": {...}}`).  
  - RAW mode or complex structured output will not be supported initially — keep it simple.  

### Directory Structure
```
.
├── README.md
├── pyproject.toml # Dependencies: fastapi, uvicorn, pydantic, llama-cpp-python, python-dotenv (optional)
├── .env.example # Sample environment variables (MODEL_PATH, N_THREADS, etc.)
├── scripts/
│ ├── dev.sh # Dev startup: uv run uvicorn app.main:app --reload
│ └── run_server.sh # Prod/test startup: uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
├── app/
│ ├── main.py # FastAPI entrypoint (router registration, middleware, /health)
│ ├── config.py # Env vars/constants (MIN_LEN, MAX_LEN, model path, etc.)
│ ├── models.py # Pydantic: Message, ChatRequest, ChatResponse
│ ├── routers/
│ │ └── chat.py # POST /chat (JSON only), calls generation with min-length control
│ ├── inference/
│ │ ├── engine.py # llama-cpp-python initialization & inference loop management (thread-safe)
│ │ ├── processors.py # MinCharLengthProcessor (EOS suppression/release, sentence-ending bias)
│ │ └── tokenizer.py # Character/token count utilities (Japanese text counting)
│ ├── utils/
│ │ ├── logging.py # Logger settings
│ │ └── text_sanitize.py # Safe trim for max length, auto-closing of brackets/quotes
│ └── schemas/
│ └── response.py # Response schema ({ "text": "...", "meta": {...} })
├── tests/
│ ├── conftest.py # Common test setup (small model/mocks for testing)
│ └── test_minlen_processor.py # Unit tests for MinCharLengthProcessor (EOS behavior before/after threshold)
└── docs/
```