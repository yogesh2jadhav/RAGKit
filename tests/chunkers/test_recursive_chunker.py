from uuid import uuid4

from ragkit.chunkers.recursive_chunker import (
    RecursiveChunker,
)
from ragkit.models.document import Document


def test_chunk_document():

    document = Document(
        id=uuid4(),
        content="A" * 2500,
    )

    chunker = RecursiveChunker(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = list(chunker.chunk(document))

    assert len(chunks) == 4

def test_invalid_overlap():

    try:

        RecursiveChunker(
            chunk_size=100,
            chunk_overlap=100,
        )

        assert False

    except ValueError:
        assert True