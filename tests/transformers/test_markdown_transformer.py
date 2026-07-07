from uuid import uuid4

from ragkit.models.document import Document
from ragkit.transformers.markdown_transformer import MarkdownTransformer


def test_remove_headers():

    transformer = MarkdownTransformer()

    document = Document(
        id=uuid4(),
        content="# Apache Spark",
        metadata={},
    )

    transformed = transformer.transform(
        document,
    )

    assert transformed.content == "Apache Spark"


def test_remove_links():

    transformer = MarkdownTransformer()

    document = Document(
        id=uuid4(),
        content="[Apache Spark](https://spark.apache.org)",
        metadata={},
    )

    transformed = transformer.transform(
        document,
    )

    assert transformed.content == "Apache Spark"


def test_remove_images():

    transformer = MarkdownTransformer()

    document = Document(
        id=uuid4(),
        content="![Logo](logo.png)",
        metadata={},
    )

    transformed = transformer.transform(
        document,
    )

    assert transformed.content == ""


def test_remove_emphasis():

    transformer = MarkdownTransformer()

    document = Document(
        id=uuid4(),
        content="**Apache** *Spark*",
        metadata={},
    )

    transformed = transformer.transform(
        document,
    )

    assert transformed.content == "Apache Spark"


def test_remove_inline_code():

    transformer = MarkdownTransformer()

    document = Document(
        id=uuid4(),
        content="Run `spark-submit` command.",
        metadata={},
    )

    transformed = transformer.transform(
        document,
    )

    assert transformed.content == "Run spark-submit command."


def test_remove_code_blocks():

    transformer = MarkdownTransformer()

    document = Document(
        id=uuid4(),
        content=(
            "Apache Spark\n\n"
            "```python\n"
            "print('hello')\n"
            "```\n"
            "Distributed engine."
        ),
        metadata={},
    )

    transformed = transformer.transform(
        document,
    )

    assert "print" not in transformed.content
    assert "Apache Spark" in transformed.content
    assert "Distributed engine." in transformed.content


def test_remove_horizontal_rule():

    transformer = MarkdownTransformer()

    document = Document(
        id=uuid4(),
        content="Apache\n---\nSpark",
        metadata={},
    )

    transformed = transformer.transform(
        document,
    )

    assert transformed.content == "Apache\n\nSpark"