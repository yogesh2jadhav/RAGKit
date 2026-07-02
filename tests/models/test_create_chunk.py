from uuid import uuid4

from ragkit.models.chunk import Chunk


def test_create_chunk() -> None:
    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        index=0,
        content="Apache Spark"
    )

    assert chunk.index == 0
    assert chunk.content == "Apache Spark"