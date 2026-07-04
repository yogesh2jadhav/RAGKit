from uuid import uuid4

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

    embeddings = list(
        embedder.embed([chunk])
    )

    assert len(embeddings) == 1

    assert len(
        embeddings[0].vector
    ) > 0