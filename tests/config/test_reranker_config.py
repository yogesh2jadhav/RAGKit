from ragkit.config.reranker_config import RerankerConfig


def test_reranker_config_defaults():

    config = RerankerConfig()

    assert config.top_k == 5
    assert config.temperature == 0.0
    assert config.max_tokens == 64