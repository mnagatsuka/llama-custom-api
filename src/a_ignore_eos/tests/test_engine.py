from src.a_ignore_eos.app.engine import generate


def test_generate_respects_min_and_max(patch_llama):
    messages = [{"role": "user", "content": "hello"}]
    out = generate(messages, min_len=16, max_len=20)
    text = out["text"]
    meta = out["meta"]
    assert 0 < len(text) <= 20
    assert meta["returned_chars"] <= 20
    assert meta["strategy"] == "ignore_eos"

