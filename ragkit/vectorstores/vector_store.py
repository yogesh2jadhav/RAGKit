"""
Purpose
-------
Defines the abstract interface for all vector store implementations.

Responsibilities
----------------
- Store chunk embeddings.
- Persist enough metadata to faithfully reconstruct a Chunk.

Does NOT
---------
- Generate embeddings.
- Perform chunking.
- Call an LLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from ragkit.models.chunk import Chunk
from ragkit.models.embedding import Embedding


class VectorStore(ABC):
    """
    Abstract base class for vector stores.

    A VectorStore persists vectors together with enough information
    to rebuild the original Chunk during retrieval.
    """

    @abstractmethod
    def add(
        self,
        chunks: Iterable[Chunk],
        embeddings: Iterable[Embedding],
    ) -> None:
        """
        Store chunks and their embeddings.

        Parameters
        ----------
        chunks
            Original chunks.

        embeddings
            Vector representations of the chunks.

        Notes
        -----
        Both iterables must represent the same logical ordering.
        Each Embedding.chunk_id must match the corresponding Chunk.id.
        """
        raise NotImplementedError