from ragkit.config import (
    EmbeddingConfig,
    LLMConfig,
    RagKitConfig,
)


def test_create_ragkit_config():
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
    assert config.retrieval.top_k == 5
    assert config.vector_store.collection_name == "ragkit"