from uuid import uuid4

from ragkit.models.chunk import Chunk
from ragkit.models.search_result import SearchResult
from ragkit.rerankers.ranking_prompt_builder import RankingPromptBuilder


def test_ranking_prompt_builder():

    builder = RankingPromptBuilder()

    results = [
        SearchResult(
            chunk=Chunk(
                id=uuid4(),
                document_id=uuid4(),
                index=0,
                content="Apache Spark",
                start_offset=0,
                end_offset=12,
                metadata={},
            ),
            score=0.9,
        ),
        SearchResult(
            chunk=Chunk(
                id=uuid4(),
                document_id=uuid4(),
                index=1,
                content="Delta Lake",
                start_offset=0,
                end_offset=10,
                metadata={},
            ),
            score=0.8,
        ),
    ]

    prompt = builder.build(
        query="What is Spark?",
        search_results=results,
    )

    assert "What is Spark?" in prompt
    assert "Apache Spark" in prompt
    assert "Delta Lake" in prompt
    assert "2,1,3" in prompt