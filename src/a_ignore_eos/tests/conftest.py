import os
import types
import pytest


@pytest.fixture(autouse=True)
def set_env_defaults(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", os.getenv("MODEL_PATH", "model.gguf"))
    monkeypatch.setenv("MIN_LEN", "16")
    monkeypatch.setenv("MAX_LEN", "64")
    # Disable second pass by default in tests
    monkeypatch.setenv("SECOND_PASS", "false")


class DummyLlama:
    def __init__(self):
        self.model_path = "dummy"

    def _stream_gen(self, text: str):
        # yield in chunks to simulate streaming
        step = 8
        for i in range(0, len(text), step):
            yield {"choices": [{"text": text[i:i+step]}]}

    def create_completion(self, prompt, max_tokens, temperature, top_p, ignore_eos=False, stream=False):
        # produce deterministic output regardless of args
        full = "A" * 40 + "。"
        if stream:
            return self._stream_gen(full)
        return {"choices": [{"text": full}], "usage": {}}


@pytest.fixture
def patch_llama(monkeypatch):
    dummy = DummyLlama()
    def fake_ensure_llama(model_path=None):
        return dummy
    monkeypatch.setattr("src.a_ignore_eos.app.engine._ensure_llama", fake_ensure_llama)
    return dummy

