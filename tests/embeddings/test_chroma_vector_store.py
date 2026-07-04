from uuid import uuid4

from ragkit.models.embedding import Embedding
from ragkit.vectorstores.chroma_vector_store import (
    ChromaVectorStore,
)


def test_store_embedding(tmp_path):

    store = ChromaVectorStore(
        path=str(tmp_path)
    )

    embedding = Embedding(
        chunk_id=uuid4(),
        content="Apache Spark",
        model="test",
        vector=[0.1, 0.2, 0.3],
    )

    store.add([embedding])

    collection = store._collection

    result = collection.get()

    assert len(result["ids"]) == 1