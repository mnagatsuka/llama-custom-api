from src.b_logit_bias.app.engine import generate


def test_generate_meta_and_trim(patch_llama):
    messages = [{"role": "user", "content": "hello"}]
    out = generate(messages, min_len=16, max_len=20)
    assert "text" in out and "meta" in out
    meta = out["meta"]
    assert meta["strategy"] == "logit_bias"
    assert meta["returned_chars"] <= 20

