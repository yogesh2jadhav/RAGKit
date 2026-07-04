from uuid import uuid4

import pytest

from ragkit.models.chunk import Chunk
from ragkit.models.embedding import Embedding
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore


def test_store_embedding(tmp_path):
    """
    Verify that a chunk and its embedding are stored.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        index=0,
        content="Apache Spark",
        start_offset=0,
        end_offset=12,
        metadata={
            "source": "unit-test",
        },
    )

    embedding = Embedding(
        chunk_id=chunk.id,
        model="nomic-embed-text",
        vector=[0.1, 0.2, 0.3],
    )

    store.add(
        chunks=[chunk],
        embeddings=[embedding],
    )

    assert store.count() == 1


def test_chunk_embedding_id_mismatch(tmp_path):
    """
    Verify that storing mismatched Chunk/Embedding pairs fails.

    This protects the database from corrupted data.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        index=0,
        content="Apache Spark",
        start_offset=0,
        end_offset=12,
        metadata={},
    )

    embedding = Embedding(
        chunk_id=uuid4(),  # Different UUID
        model="nomic-embed-text",
        vector=[0.1, 0.2],
    )

    with pytest.raises(
        ValueError,
        match="Chunk.id does not match Embedding.chunk_id.",
    ):
        store.add(
            chunks=[chunk],
            embeddings=[embedding],
        )