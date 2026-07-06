from uuid import uuid4

from ragkit.models.document import Document
from ragkit.transformers.identity_transformer import IdentityTransformer


def test_identity_transformer_returns_same_document():

    document = Document(
        id=uuid4(),
        content="Apache Spark",
        metadata={
            "source": "spark.txt",
        },
    )

    transformer = IdentityTransformer()

    transformed = transformer.transform(
        document,
    )

    assert transformed is document