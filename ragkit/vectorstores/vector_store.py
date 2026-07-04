"""
Purpose
-------
Defines the abstract interface for all vector store implementations.

Responsibilities
----------------
- Store chunk embeddings.
- Search for similar chunks.

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
from ragkit.models.query_embedding import QueryEmbedding
from ragkit.models.search_result import SearchResult

# This is like interface in java
class VectorStore(ABC):
    """
    Abstract base class for all vector stores.
    """

    @abstractmethod
    def add(
        self,
        chunks: Iterable[Chunk],
        embeddings: Iterable[Embedding],
    ) -> None:
        """
        Store chunks together with their embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: QueryEmbedding,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Search the vector store.

        Parameters
        ----------
        query_embedding
            Embedding generated from the user's question.

        top_k
            Maximum number of similar chunks to return.

        Returns
        -------
        list[SearchResult]
            Search results ordered by descending similarity.
        """
        raise NotImplementedError