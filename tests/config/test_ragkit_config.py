from ragkit.config import (
    EmbeddingConfig,
    LLMConfig,
    RagKitConfig,
    RetrievalConfig,
    VectorStoreConfig,
)


def test_create_ragkit_config():
    """
    Verify RagKitConfig stores all configuration objects.
    """

    config = RagKitConfig(
        embedding=EmbeddingConfig(
            model="nomic-embed-text",
        ),
        llm=LLMConfig(
            model="qwen3:8b",
        ),
    )

    assert config.embedding.model == "nomic-embed-text"

    assert config.llm.model == "qwen3:8b"

    assert isinstance(
        config.retrieval,
        RetrievalConfig,
    )

    assert isinstance(
        config.vector_store,
        VectorStoreConfig,
    )

    assert config.retrieval.top_k == 5

    assert config.vector_store.collection_name == "ragkit"