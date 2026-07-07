from uuid import uuid4

from ragkit.config.reranker_config import RerankerConfig
from ragkit.llms.llm import LLM
from ragkit.models.chunk import Chunk
from ragkit.models.llm_response import LLMResponse
from ragkit.models.search_result import SearchResult
from ragkit.rerankers.llm_reranker import LLMReranker


class FakeLLM(LLM):

    def __init__(self):

        self.called = False

    def generate(
        self,
        prompt: str,
        options=None,
    ) -> LLMResponse:

        self.called = True

        return LLMResponse(
            content="1",
        )


def test_llm_reranker_returns_results():

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
                content="Apache Spark",
                start_offset=0,
                end_offset=12,
                metadata={},
            ),
            score=0.9,
        )
    ]

    reranked = reranker.rerank(
        query="What is Spark?",
        results=results,
    )

    assert reranked == results