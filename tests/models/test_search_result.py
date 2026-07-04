from uuid import uuid4

from ragkit.models.chunk import Chunk
from ragkit.models.search_result import SearchResult


def test_create_search_result():
    """
    Verify that SearchResult stores
    the retrieved Chunk and similarity score.
    """

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

    result = SearchResult(
        chunk=chunk,
        similarity=0.987,
    )

    assert result.chunk == chunk
    assert result.similarity == 0.987