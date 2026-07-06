from ragkit.config.embedding_config import EmbeddingConfig


def test_create_embedding_config():
    config = EmbeddingConfig(
        model="nomic-embed-text",
    )

    assert config.model == "nomic-embed-text"