from src.c_logits_processor.app.processors import MinCharLengthProcessor


def test_minlen_processor_suppresses_eos():
    proc = MinCharLengthProcessor(eos_token_id=5, min_len=10)
    # Before reaching min, EOS should be -inf
    logits = [0.0] * 10
    logits = proc(logits, [])
    assert logits[5] == float('-inf')


def test_minlen_processor_releases_after_threshold():
    proc = MinCharLengthProcessor(eos_token_id=5, min_len=3, punctuation_token_ids=[7], punctuation_bias=0.5)
    proc.update_char_count("abc")
    logits = [0.0] * 10
    logits = proc(logits, [])
    assert logits[5] != float('-inf')
    assert logits[7] > 0.0


def test_engine_generate_uses_trim_and_meta(patch_llama):
    from src.c_logits_processor.app.engine import generate
    messages = [{"role": "user", "content": "hello"}]
    result = generate(messages, min_len=16, max_len=20)
    assert "text" in result and "meta" in result
    assert result["meta"]["returned_chars"] <= 20
