from ragkit.app import RagKit
from ragkit.indexers.indexer import Indexer
from ragkit.models.indexing_result import IndexingResult
from ragkit.models.llm_response import LLMResponse
from ragkit.pipelines.pipeline import Pipeline
from ragkit.sources.source import Source


class FakeIndexer(Indexer):

    def __init__(self):

        self.called = False

    def index(
        self,
        source: Source,
    ) -> IndexingResult:

        self.called = True

        return IndexingResult(
            documents=1,
            chunks=2,
            embeddings=2,
        )


class FakePipeline(Pipeline):

    def __init__(self):

        self.called = False

    def invoke(
        self,
        query: str,
    ) -> LLMResponse:

        self.called = True

        return LLMResponse(
            content="Apache Spark is a distributed engine.",
        )


def test_index():

    indexer = FakeIndexer()

    rag = RagKit(
        indexer=indexer,
        pipeline=FakePipeline(),
    )

    result = rag.index(None)

    assert indexer.called

    assert result.documents == 1
    assert result.chunks == 2
    assert result.embeddings == 2


def test_query():

    pipeline = FakePipeline()

    rag = RagKit(
        indexer=FakeIndexer(),
        pipeline=pipeline,
    )

    response = rag.query(
        "What is Spark?",
    )

    assert pipeline.called

    assert response.content == (
        "Apache Spark is a distributed engine."
    )