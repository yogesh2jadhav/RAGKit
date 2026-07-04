"""
Purpose
-------
Generates embeddings using a SentenceTransformer model.

Responsibilities
----------------
- Load the embedding model.
- Generate vector embeddings.
- Return Embedding domain objects.

Does NOT
--------
- Store embeddings.
- Query vector databases.
"""

from __future__ import annotations

from collections.abc import Iterable

from sentence_transformers import SentenceTransformer

from ragkit.embeddings.embedder import Embedder
from ragkit.models.chunk import Chunk
from ragkit.models.embedding import Embedding


class SentenceTransformerEmbedder(Embedder):

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2", # This is model name all-MiniLM-L6-v2 produces 384-dimensional vectors.
    ) -> None:

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(
        self,
        chunks: Iterable[Chunk],
    ) -> Iterable[Embedding]:

        chunk_list = list(chunks)  # Create list of chunks

        texts = [ # chunk have other information also from that only take text in "texts" object. But we are collecting only "content"
            chunk.content
            for chunk in chunk_list
        ]

        vectors = self._model.encode( # All list of texts send to convert into vectors
            texts,
            convert_to_numpy=True,
        )
        '''
         Zip is createing pair of Chuck1 <-> vector1, Chunk2 <-> vector2
         Input to Zip
             chunk_list = ["Spark", "Delta", "Databricks"]
             vectors = [95, 88, 91]
         Output of Zip
            Spark 95
            Delta 88
            Databricks 91
        '''
        for chunk, vector in zip(
            chunk_list,
            vectors,
            strict=True,
        ):
            yield Embedding(
                chunk_id=chunk.id,
                content=chunk.content,
                model=self._model_name,
                vector=vector.tolist(),
            )