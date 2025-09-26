import os
import types
import pytest


@pytest.fixture(autouse=True)
def set_env_defaults(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", os.getenv("MODEL_PATH", "model.gguf"))
    monkeypatch.setenv("MIN_LEN", "16")
    monkeypatch.setenv("MAX_LEN", "64")


class DummyLlama:
    def __init__(self):
        self.model_path = "dummy"

    def tokenize(self, s, add_bos=False, special=False):
        # Fake token ids: map char to ord modulo a small range
        return [min(255, ord(s[0]))] if s else []

    def create_completion(self, prompt, max_tokens, temperature, top_p, logits_processor, stream=False):
        # Return a deterministic string of a certain length
        text = "A" * 40 + "。"
        return {"choices": [{"text": text}], "usage": {"prompt_tokens": 0, "completion_tokens": len(text)}}

    def token_eos(self):
        return 2


@pytest.fixture
def patch_llama(monkeypatch):
    dummy = DummyLlama()
    def fake_ensure_llama(model_path=None):
        return dummy
    monkeypatch.setattr(
        "src.c_logits_processor.app.engine._ensure_llama",
        fake_ensure_llama,
    )
    return dummy
