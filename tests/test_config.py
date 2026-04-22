from sparsefire.config import Config


def test_config_defaults():
    c = Config()
    assert c.model_id == "meta-llama/Llama-3.2-1B-Instruct"
    assert c.attn_impl == "eager"
    assert c.n_tokens == 256
    assert c.n_runs == 50


def test_config_override_preserves_frozen():
    c = Config()
    c2 = c.override(n_runs=10, n_tokens=64)
    assert c.n_runs == 50 and c2.n_runs == 10
    assert c2.n_tokens == 64
