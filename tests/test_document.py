from uuid import uuid4

from ragkit.models.document import Document


def test_create_document() -> None:
    document = Document(
        id=uuid4(),
        content="Apache Spark",
    )

    assert document.content == "Apache Spark"
    assert document.metadata == {}