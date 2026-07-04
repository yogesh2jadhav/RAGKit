from uuid import uuid4

from ragkit.models.chunk import Chunk
from ragkit.models.embedding import Embedding
from ragkit.models.query_embedding import QueryEmbedding
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore


def create_chunk(content: str, index: int = 0) -> Chunk:
    """
    Create a Chunk for testing.
    """
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        index=index,
        content=content,
        start_offset=0,
        end_offset=len(content),
        metadata={
            "source": "unit-test",
        },
    )


def create_embedding(chunk: Chunk, vector: list[float]) -> Embedding:
    """
    Create an Embedding for testing.
    """
    return Embedding(
        chunk_id=chunk.id,
        model="unit-test-model",
        vector=vector,
    )


def test_store_embedding(tmp_path):
    """
    Verify a chunk and embedding are stored.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    chunk = create_chunk("Apache Spark")

    embedding = create_embedding(
        chunk,
        [0.1, 0.2, 0.3],
    )

    store.add(
        chunks=[chunk],
        embeddings=[embedding],
    )

    assert store.count() == 1


def test_missing_embedding_for_chunk(tmp_path):
    """
    Verify missing embeddings are rejected.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    chunk = create_chunk("Apache Spark")

    embedding = Embedding(
        chunk_id=uuid4(),
        model="unit-test-model",
        vector=[0.1, 0.2],
    )

    try:
        store.add(
            chunks=[chunk],
            embeddings=[embedding],
        )
        assert False
    except ValueError:
        pass


def test_store_embeddings_in_any_order(tmp_path):
    """
    Verify ordering does not matter.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    chunk1 = create_chunk("Chunk 1", 0)
    chunk2 = create_chunk("Chunk 2", 1)

    embedding1 = create_embedding(
        chunk1,
        [0.1, 0.2],
    )

    embedding2 = create_embedding(
        chunk2,
        [0.3, 0.4],
    )

    store.add(
        chunks=[chunk1, chunk2],
        embeddings=[embedding2, embedding1],
    )

    assert store.count() == 2


def test_search_empty_database(tmp_path):
    """
    Searching an empty database should return no results.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    query = QueryEmbedding(
        model="unit-test-model",
        vector=[0.1, 0.2],
    )

    results = list(
        store.search(
            query_embedding=query,
            top_k=5,
        )
    )

    assert results == []


def test_search_returns_chunk(tmp_path):
    """
    Verify a stored chunk can be retrieved.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    chunk = create_chunk("Apache Spark")

    embedding = create_embedding(
        chunk,
        [0.1, 0.2, 0.3],
    )

    store.add(
        chunks=[chunk],
        embeddings=[embedding],
    )

    query = QueryEmbedding(
        model="unit-test-model",
        vector=[0.1, 0.2, 0.3],
    )

    results = list(
        store.search(
            query_embedding=query,
            top_k=1,
        )
    )

    assert len(results) == 1

    result = results[0]

    assert result.chunk.content == chunk.content
    assert result.chunk.document_id == chunk.document_id
    assert result.chunk.index == chunk.index
    assert result.chunk.metadata == chunk.metadata


def test_search_top_k(tmp_path):
    """
    Verify top_k limits the number of returned results.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    chunks = []

    embeddings = []

    for i in range(5):

        chunk = create_chunk(
            f"Chunk {i}",
            i,
        )

        chunks.append(chunk)

        embeddings.append(
            create_embedding(
                chunk,
                [float(i), float(i + 1)],
            )
        )

    store.add(
        chunks=chunks,
        embeddings=embeddings,
    )

    query = QueryEmbedding(
        model="unit-test-model",
        vector=[0.0, 1.0],
    )

    results = list(
        store.search(
            query_embedding=query,
            top_k=2,
        )
    )

    assert len(results) == 2