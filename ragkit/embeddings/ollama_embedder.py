"""
Purpose
-------
Generate vector embeddings using a locally running Ollama embedding model.

Responsibilities
----------------
- Connect to Ollama.
- Generate embeddings for chunks.
- Convert Ollama responses into Embedding domain objects.

Does NOT
--------
- Store embeddings.
- Query vector databases.
- Generate LLM responses.
"""

from __future__ import annotations

from collections.abc import Iterable

import ollama

from ragkit.embeddings.embedder import Embedder
from ragkit.models.chunk import Chunk
from ragkit.models.embedding import Embedding


class OllamaEmbedder(Embedder):
    """
    Generates embeddings using Ollama.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text", #nomic-embed-text is currently 768 dimensions.
    ) -> None:
        """
        Initialize the embedder.

        Parameters
        ----------
        model:
            Name of the Ollama embedding model.
        """
        self._model = model

    def embed(
        self,
        chunks: Iterable[Chunk],
    ) -> Iterable[Embedding]:
        """
        Generate embeddings for every supplied chunk.

        The implementation performs one request per chunk.
        In a future version we'll batch requests for better performance.
        """

        for chunk in chunks:

            response = ollama.embed(
                model=self._model,
                input=chunk.content,
            )

            vector = response["embeddings"][0]

            yield Embedding(
                chunk_id=chunk.id,
                model=self._model,
                vector=vector,
            )