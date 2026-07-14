from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from ragkit.keyword.keyword_searcher import KeywordSearcher
from ragkit.models.chunk import Chunk
from ragkit.models.search_result import SearchResult
from ragkit.retrievers.hybrid_retriever import HybridRetriever
from ragkit.retrievers.retriever import Retriever


class FakeRetriever(Retriever):
    """
    Fake retriever used for testing.
    """

    def __init__(
        self,
        results: list[SearchResult],
    ) -> None:
        self.results = results
        self.last_query = None
        self.last_top_k = None

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        self.last_query = query
        self.last_top_k = top_k
        return self.results


class FakeKeywordSearcher(KeywordSearcher):
    """
    Fake keyword searcher used for testing.
    """

    def __init__(
        self,
        results: list[SearchResult],
    ) -> None:
        self.results = results
        self.last_query = None
        self.last_top_k = None

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> Iterable[SearchResult]:
        self.last_query = query
        self.last_top_k = top_k
        yield from self.results


def create_result(
    content: str,
) -> SearchResult:
    """
    Create a SearchResult for testing.
    """
    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        index=0,
        content=content,
        start_offset=0,
        end_offset=len(content),
        metadata={},
    )
    return SearchResult(
        chunk=chunk,
        score=1.0,
    )


def test_hybrid_retriever_merges_results():
    """
    Verify semantic and keyword results are merged.
    """
    semantic = [
        create_result(
            "Apache Spark",
        ),
    ]

    keyword = [
        create_result(
            "Delta Lake",
        ),
    ]

    retriever = HybridRetriever(
        retriever=FakeRetriever(
            semantic,
        ),
        keyword_searcher=FakeKeywordSearcher(
            keyword,
        ),
    )

    results = retriever.retrieve(
        "spark",
    )

    assert len(results) == 2

    assert results[0].chunk.content == "Apache Spark"

    assert results[1].chunk.content == "Delta Lake"


def test_hybrid_retriever_removes_duplicates():
    """
    Verify duplicate chunks are removed.
    """

    shared = create_result(
        "Apache Spark",
    )

    retriever = HybridRetriever(
        retriever=FakeRetriever(
            [shared],
        ),
        keyword_searcher=FakeKeywordSearcher(
            [shared],
        ),
    )

    results = retriever.retrieve(
        "spark",
    )

    assert len(results) == 1


def test_hybrid_retriever_respects_top_k():
    """
    Verify top_k limits returned results.
    """

    semantic = [
        create_result("A"),
        create_result("B"),
        create_result("C"),
    ]

    keyword = [
        create_result("D"),
        create_result("E"),
    ]

    retriever = HybridRetriever(
        retriever=FakeRetriever(
            semantic,
        ),
        keyword_searcher=FakeKeywordSearcher(
            keyword,
        ),
    )

    results = retriever.retrieve(
        "test",
        top_k=3,
    )

    assert len(results) == 3


def test_hybrid_retriever_passes_query():
    """
    Verify query is passed to both retrieval strategies.
    """

    retriever_impl = FakeRetriever([])

    keyword = FakeKeywordSearcher([])

    retriever = HybridRetriever(
        retriever=retriever_impl,
        keyword_searcher=keyword,
    )

    retriever.retrieve(
        "Apache Spark",
    )

    assert retriever_impl.last_query == "Apache Spark"

    assert keyword.last_query == "Apache Spark"


def test_hybrid_retriever_passes_top_k():
    """
    Verify top_k is forwarded to both retrieval strategies.
    """

    retriever_impl = FakeRetriever([])

    keyword = FakeKeywordSearcher([])

    retriever = HybridRetriever(
        retriever=retriever_impl,
        keyword_searcher=keyword,
    )

    retriever.retrieve(
        "Apache Spark",
        top_k=7,
    )

    assert retriever_impl.last_top_k == 7

    assert keyword.last_top_k == 7