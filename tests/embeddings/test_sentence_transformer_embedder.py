from uuid import uuid4

from ragkit.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from ragkit.models.chunk import Chunk


def test_embed_chunk():

    embedder = SentenceTransformerEmbedder()

    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        index=0,
        start_offset=0,
        end_offset=10,
        content="Apache Spark",
    )

    embeddings = list(
        embedder.embed([chunk])
    )

    assert len(embeddings) == 1

    assert len(
        embeddings[0].vector
    ) > 0