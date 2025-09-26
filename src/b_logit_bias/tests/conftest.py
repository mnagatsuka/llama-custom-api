import os
import pytest


@pytest.fixture(autouse=True)
def set_env_defaults(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", os.getenv("MODEL_PATH", "model.gguf"))
    monkeypatch.setenv("MIN_LEN", "16")
    monkeypatch.setenv("MAX_LEN", "64")
    monkeypatch.setenv("EOS_BIAS", "-10.0")
    monkeypatch.setenv("SECOND_PASS", "false")


class DummyLlama:
    def __init__(self):
        self.model_path = "dummy"

    def token_eos(self):
        return 2

    def _stream_gen(self, text: str):
        step = 8
        for i in range(0, len(text), step):
            yield {"choices": [{"text": text[i:i+step]}]}

    def create_completion(self, prompt, max_tokens, temperature, top_p, logit_bias=None, stream=False):
        full = "B" * 40 + "."
        if stream:
            return self._stream_gen(full)
        return {"choices": [{"text": full}], "usage": {}}


@pytest.fixture
def patch_llama(monkeypatch):
    dummy = DummyLlama()

    def fake_ensure_llama(model_path=None):
        return dummy

    # patch ensure
    monkeypatch.setattr("src.b_logit_bias.app.engine._ensure_llama", fake_ensure_llama)
    # ensure eos id on import path
    monkeypatch.setattr("src.b_logit_bias.app.engine._eos_id", 2, raising=False)
    return dummy

