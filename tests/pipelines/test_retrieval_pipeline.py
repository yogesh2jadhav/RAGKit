from collections.abc import Iterable
from uuid import uuid4

from ragkit.llms.llm import LLM
from ragkit.models.chunk import Chunk
from ragkit.models.llm_response import LLMResponse
from ragkit.models.search_result import SearchResult
from ragkit.pipelines.retrieval_pipeline import RetrievalPipeline
from ragkit.prompts.prompt_builder import PromptBuilder
from ragkit.rerankers.reranker import Reranker
from ragkit.retrievers.retriever import Retriever


class FakeRetriever(Retriever):
    """
    Fake Retriever used for unit testing.
    """

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> Iterable[SearchResult]:

        yield SearchResult(
            chunk=Chunk(
                id=uuid4(),
                document_id=uuid4(),
                index=0,
                content="Apache Spark is a distributed engine.",
                start_offset=0,
                end_offset=36,
                metadata={},
            ),
            score=0.1,
        )


class FakeReranker(Reranker):
    """
    Fake Reranker used for unit testing.
    """

    def __init__(self):

        self.called = False

    def rerank(
        self,
        query: str,
        results: Iterable[SearchResult],
    ) -> list[SearchResult]:

        self.called = True

        return list(results)


class FakePromptBuilder(PromptBuilder):
    """
    Fake PromptBuilder used for unit testing.
    """

    def build(
        self,
        query: str,
        search_results: Iterable[SearchResult],
    ) -> str:

        return "Prompt"


class FakeLLM(LLM):
    """
    Fake LLM used for unit testing.
    """

    def generate(
        self,
        prompt: str,
        options=None,
    ) -> LLMResponse:

        return LLMResponse(
            content="Answer",
        )


def test_retrieval_pipeline():
    """
    Verify RetrievalPipeline orchestrates all
    pipeline components.
    """

    reranker = FakeReranker()

    pipeline = RetrievalPipeline(
        retriever=FakeRetriever(),
        reranker=reranker,
        prompt_builder=FakePromptBuilder(),
        llm=FakeLLM(),
    )

    response = pipeline.invoke(
        query="What is Spark?",
    )

    assert reranker.called

    assert response.content == "Answer"