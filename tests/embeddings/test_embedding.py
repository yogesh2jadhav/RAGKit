from uuid import uuid4

from ragkit.models.embedding import Embedding


def test_create_embedding():

    embedding = Embedding(
        chunk_id=uuid4(),
        model="test-model",
        vector=[0.1, 0.2, 0.3],
    )

    assert embedding.model == "test-model"

    assert len(embedding.vector) == 3