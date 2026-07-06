from ragkit.config.retrieval_config import RetrievalConfig


def test_create_retrieval_config():
    config = RetrievalConfig()

    assert config.top_k == 5