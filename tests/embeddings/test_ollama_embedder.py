from unittest.mock import Mock
from uuid import uuid4

from ragkit.config.embedding_config import EmbeddingConfig
from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.models.chunk import Chunk


def test_generate_embedding():

    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        index=0,
        start_offset=0,
        end_offset=10,
        content="Apache Spark",
    )

    embedder = OllamaEmbedder()

    #
    # Mock Ollama response
    #
    mock_response = Mock()
    mock_response.embeddings = [
        [0.1, 0.2, 0.3]
    ]

    embedder._client.embed = Mock(
        return_value=mock_response
    )

    embeddings = list(
        embedder.embed([chunk])
    )

    assert len(embeddings) == 1

    assert embeddings[0].vector == [
        0.1,
        0.2,
        0.3,
    ]

    embedder._client.embed.assert_called_once()


def test_create_embedder_with_default_model():
    """
    Verify the default embedding model is used.
    """

    embedder = OllamaEmbedder()

    assert embedder._model_name == "nomic-embed-text"


def test_create_embedder_with_config():
    """
    Verify EmbeddingConfig overrides the default model.
    """

    config = EmbeddingConfig(
        model="mxbai-embed-large",
    )

    embedder = OllamaEmbedder(
        config=config,
    )

    assert embedder._model_name == "mxbai-embed-large"
