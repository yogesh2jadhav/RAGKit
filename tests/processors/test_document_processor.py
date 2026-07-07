from collections.abc import Iterable
from uuid import uuid4

from ragkit.chunkers.chunker import Chunker
from ragkit.embeddings.embedder import Embedder
from ragkit.models.chunk import Chunk
from ragkit.models.document import Document
from ragkit.models.embedding import Embedding
from ragkit.models.query_embedding import QueryEmbedding
from ragkit.processors.document_processor import DocumentProcessor
from ragkit.transformers.transformer import Transformer


class FakeTransformer(Transformer):
    """
    Fake Transformer used for unit testing.
    """

    def __init__(self) -> None:

        self.called = False

    def transform(
        self,
        document: Document,
    ) -> Document:

        self.called = True

        return document


class FakeChunker(Chunker):
    """
    Fake Chunker used for unit testing.
    """

    def __init__(self) -> None:

        self.called = False

    def chunk(
        self,
        document: Document,
    ) -> Iterable[Chunk]:

        self.called = True

        yield Chunk(
            id=uuid4(),
            document_id=document.id,
            index=0,
            content=document.content,
            start_offset=0,
            end_offset=len(document.content),
            metadata=document.metadata,
        )


class FakeEmbedder(Embedder):
    """
    Fake Embedder used for unit testing.
    """

    def __init__(self) -> None:

        self.called = False

    def embed(
        self,
        chunks: Iterable[Chunk],
    ) -> Iterable[Embedding]:

        self.called = True

        for chunk in chunks:

            yield Embedding(
                chunk_id=chunk.id,
                model="unit-test",
                vector=[0.1, 0.2, 0.3],
            )

    def embed_query(
        self,
        query: str,
    ) -> QueryEmbedding:

        return QueryEmbedding(
            model="unit-test",
            vector=[0.1, 0.2, 0.3],
        )


def test_document_processor_processes_document():
    """
    Verify DocumentProcessor orchestrates
    transformation, chunking and embedding.
    """

    transformer = FakeTransformer()
    chunker = FakeChunker()
    embedder = FakeEmbedder()

    processor = DocumentProcessor(
        transformer=transformer,
        chunker=chunker,
        embedder=embedder,
    )

    document = Document(
        id=uuid4(),
        content="Apache Spark",
        metadata={
            "source": "spark.txt",
        },
    )

    chunks, embeddings = processor.process(
        document,
    )

    assert transformer.called
    assert chunker.called
    assert embedder.called

    assert len(chunks) == 1
    assert len(embeddings) == 1

    assert chunks[0].content == "Apache Spark"
    assert embeddings[0].chunk_id == chunks[0].id