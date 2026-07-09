from uuid import uuid4

from ragkit.keyword.bm25_searcher import BM25Searcher
from ragkit.models.chunk import Chunk
from ragkit.models.embedding import Embedding
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore


def create_chunk(
    content: str,
    index: int = 0,
) -> Chunk:
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        index=index,
        content=content,
        start_offset=0,
        end_offset=len(content),
        metadata={},
    )


def create_embedding(
    chunk: Chunk,
) -> Embedding:

    #
    # BM25 never uses embeddings,
    # but Chroma requires one.
    #
    return Embedding(
        chunk_id=chunk.id,
        model="unit-test",
        vector=[0.1, 0.2, 0.3],
    )


def test_bm25_returns_best_match(
    tmp_path,
):
    """
    Verify BM25 returns the best keyword match.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    chunks = [
        create_chunk("Apache Spark is fast"),
        create_chunk("Python is easy"),
        create_chunk("Delta Lake supports ACID"),
    ]

    store.add(
        chunks=chunks,
        embeddings=[
            create_embedding(chunk)
            for chunk in chunks
        ],
    )

    searcher = BM25Searcher(
        vector_store=store,
    )

    results = list(
        searcher.search(
            "spark",
        )
    )

    assert len(results) > 0

    assert (
        results[0]
        .chunk
        .content
        == "Apache Spark is fast"
    )


def test_bm25_respects_top_k(
    tmp_path,
):
    """
    Verify top_k limits results.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    chunks = [
        create_chunk(f"Spark document {i}")
        for i in range(5)
    ]

    store.add(
        chunks=chunks,
        embeddings=[
            create_embedding(chunk)
            for chunk in chunks
        ],
    )

    searcher = BM25Searcher(
        vector_store=store,
    )

    results = list(
        searcher.search(
            "spark",
            top_k=2,
        )
    )

    assert len(results) == 2


def test_bm25_unknown_query(
    tmp_path,
):
    """
    Verify unknown queries do not fail.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    chunk = create_chunk(
        "Apache Spark",
    )

    store.add(
        chunks=[chunk],
        embeddings=[
            create_embedding(chunk)
        ],
    )

    searcher = BM25Searcher(
        vector_store=store,
    )

    results = list(
        searcher.search(
            "abcdefghijk",
        )
    )

    #
    # BM25 returns a score even if it is zero.
    #
    assert len(results) == 1