from uuid import uuid4

import pytest

from ragkit.models.chunk import Chunk
from ragkit.models.embedding import Embedding
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore
from ragkit.models.query_embedding import QueryEmbedding
from ragkit.models.search_result import SearchResult

def test_store_embedding(tmp_path):
    """
    Verify that a chunk and its embedding are stored successfully.
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


def test_missing_embedding_for_chunk(tmp_path):
    """
    Verify that a chunk without a matching embedding
    raises a descriptive ValueError.
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
        match="No embedding found for chunk",
    ):
        store.add(
            chunks=[chunk],
            embeddings=[embedding],
        )

def test_store_embeddings_in_any_order(tmp_path):
    """
    Embeddings can be supplied in any order because they
    are matched using chunk_id.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    chunk1 = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        index=0,
        content="Chunk 1",
        start_offset=0,
        end_offset=7,
        metadata={},
    )

    chunk2 = Chunk(
        id=uuid4(),
        document_id=chunk1.document_id,
        index=1,
        content="Chunk 2",
        start_offset=8,
        end_offset=15,
        metadata={},
    )

    embedding1 = Embedding(
        chunk_id=chunk1.id,
        model="nomic-embed-text",
        vector=[0.1, 0.2],
    )

    embedding2 = Embedding(
        chunk_id=chunk2.id,
        model="nomic-embed-text",
        vector=[0.3, 0.4],
    )

    # Intentionally reverse the embeddings.
    store.add(
        chunks=[chunk1, chunk2],
        embeddings=[embedding2, embedding1],
    )

    assert store.count() == 2
    def search(
        self,
        query_embedding: QueryEmbedding,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Search for the most similar chunks.

        Responsibilities
        ----------------
        - This is currently a placeholder implementation.
        - It exists so the project remains buildable while the
          retrieval feature is being developed.

        Does NOT
        --------
        - Perform an actual similarity search.
        """

        raise NotImplementedError(
            "ChromaVectorStore.search() has not been implemented yet."
        )