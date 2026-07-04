from collections.abc import Iterable

from ragkit.llms.llm import LLM
from ragkit.models.chunk import Chunk
from ragkit.models.llm_response import LLMResponse
from ragkit.models.search_result import SearchResult
from ragkit.pipelines.retrieval_pipeline import RetrievalPipeline
from ragkit.prompts.prompt_builder import PromptBuilder
from ragkit.retrievers.retriever import Retriever
from uuid import uuid4


class FakeRetriever(Retriever):

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
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


class FakePromptBuilder(PromptBuilder):

    def build(
        self,
        query: str,
        search_results: Iterable[SearchResult],
    ) -> str:

        return "Prompt"


class FakeLLM(LLM):

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
    Verify the RetrievalPipeline orchestrates the
    Retriever, PromptBuilder and LLM.
    """

    pipeline = RetrievalPipeline(
        retriever=FakeRetriever(),
        prompt_builder=FakePromptBuilder(),
        llm=FakeLLM(),
    )

    response = pipeline.invoke(
        query="What is Spark?",
    )

    assert response.content == "Answer"