from uuid import uuid4

from ragkit.config.reranker_config import RerankerConfig
from ragkit.llms.llm import LLM
from ragkit.models.chunk import Chunk
from ragkit.models.llm_response import LLMResponse
from ragkit.models.search_result import SearchResult
from ragkit.rerankers.llm_reranker import LLMReranker


class FakeLLM(LLM):
    """
    Fake LLM used for unit testing.
    """

    def __init__(self):
        self.called = False

    def generate(
        self,
        prompt: str,
        options=None,
    ) -> LLMResponse:

        self.called = True

        #
        # Swap the first two results.
        #
        return LLMResponse(
            content="2,1",
        )


def test_llm_reranker_reorders_results():

    llm = FakeLLM()

    reranker = LLMReranker(
        llm=llm,
        config=RerankerConfig(),
    )

    results = [
        SearchResult(
            chunk=Chunk(
                id=uuid4(),
                document_id=uuid4(),
                index=0,
                content="First",
                start_offset=0,
                end_offset=5,
                metadata={},
            ),
            score=0.9,
        ),
        SearchResult(
            chunk=Chunk(
                id=uuid4(),
                document_id=uuid4(),
                index=1,
                content="Second",
                start_offset=0,
                end_offset=6,
                metadata={},
            ),
            score=0.8,
        ),
    ]

    reranked = reranker.rerank(
        query="test",
        results=results,
    )

    assert llm.called

    assert reranked[0].chunk.content == "Second"
    assert reranked[1].chunk.content == "First"