from uuid import uuid4

from ragkit.models.document import Document
from ragkit.transformers.whitespace_transformer import (
    WhitespaceTransformer,
)


def test_collapse_multiple_spaces():

    transformer = WhitespaceTransformer()

    document = Document(
        id=uuid4(),
        content="Apache     Spark",
        metadata={},
    )

    transformed = transformer.transform(
        document,
    )

    assert transformed.content == "Apache Spark"


def test_collapse_blank_lines():

    transformer = WhitespaceTransformer()

    document = Document(
        id=uuid4(),
        content="Apache\n\n\n\nSpark",
        metadata={},
    )

    transformed = transformer.transform(
        document,
    )

    assert transformed.content == "Apache\n\nSpark"


def test_trim_document():

    transformer = WhitespaceTransformer()

    document = Document(
        id=uuid4(),
        content="   Apache Spark   ",
        metadata={},
    )

    transformed = transformer.transform(
        document,
    )

    assert transformed.content == "Apache Spark"


def test_preserve_single_blank_line():

    transformer = WhitespaceTransformer()

    document = Document(
        id=uuid4(),
        content="Apache\n\nSpark",
        metadata={},
    )

    transformed = transformer.transform(
        document,
    )

    assert transformed.content == "Apache\n\nSpark"