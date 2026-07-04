"""
Purpose
-------
Generate vector embeddings using a locally running Ollama embedding model.

Responsibilities
----------------
- Connect to Ollama.
- Generate embeddings for Chunk objects.
- Return Embedding domain objects.

Does NOT
--------
- Store embeddings.
- Perform similarity search.
- Generate chat responses.
"""

from __future__ import annotations

from collections.abc import Iterable

from ollama import Client

from ragkit.embeddings.embedder import Embedder
from ragkit.models.chunk import Chunk
from ragkit.models.embedding import Embedding


class OllamaEmbedder(Embedder):
    """
    Generates embeddings using an Ollama embedding model.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
    ) -> None:
        """
        Parameters
        ----------
        model
            Ollama embedding model.

        host
            Ollama server URL.
        """

        self._model = model
        self._client = Client(host=host)

    def embed(
        self,
        chunks: Iterable[Chunk],
    ) -> Iterable[Embedding]:
        """
        Generate embeddings for supplied chunks.

        Notes
        -----
        Version 1 performs one request per chunk.

        Version 2 will batch multiple chunks into a
        single Ollama request.
        """

        for chunk in chunks:

            response = self._client.embed(
                model=self._model,
                input=chunk.content,
            )

            vector = response.embeddings[0]

            yield Embedding(
                chunk_id=chunk.id,
                content=chunk.content,
                model=self._model,
                vector=vector,
            )