from uuid import uuid4

from ragkit.models.chunk import Chunk
from ragkit.models.search_result import SearchResult
from ragkit.rerankers.identity_reranker import IdentityReranker


def test_identity_reranker_returns_same_results():

    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        index=0,
        content="Apache Spark",
        start_offset=0,
        end_offset=12,
    )

    results = [
        SearchResult(
            chunk=chunk,
            score=0.42,
        )
    ]

    reranker = IdentityReranker()

    reranked = reranker.rerank(
        query="What is Spark?",
        results=results,
    )

    assert reranked == results