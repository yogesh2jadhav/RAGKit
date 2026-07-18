from uuid import uuid4

import ragkit.embeddings.sentence_transformer_embedder as sentence_transformer_module
from ragkit.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from ragkit.models.chunk import Chunk


class FakeVector(list):
    def tolist(self):
        return list(self)


class FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(self, texts, convert_to_numpy=True):
        if isinstance(texts, str):
            return FakeVector([0.1, 0.2, 0.3])

        return [
            FakeVector([0.1, 0.2, 0.3])
            for _ in texts
        ]


def test_embed_chunk(monkeypatch):

    monkeypatch.setattr(
        sentence_transformer_module,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = SentenceTransformerEmbedder()

    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        index=0,
        start_offset=0,
        end_offset=10,
        content="Apache Spark",
    )

    embeddings = list(
        embedder.embed([chunk])
    )

    assert len(embeddings) == 1

    assert len(
        embeddings[0].vector
    ) > 0
