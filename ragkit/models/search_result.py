"""
Purpose
-------
Represents the result returned by a vector similarity search.

Responsibilities
----------------
- Hold the retrieved Chunk.
- Hold the retrieval score returned by the vector store.

Does NOT
--------
- Perform vector search.
- Rank results.
- Generate embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragkit.models.chunk import Chunk


@dataclass(frozen=True, slots=True)
class SearchResult:
    """
    Represents a single search result returned by a VectorStore.

    Attributes
    ----------
    chunk
        The retrieved chunk.

    score
        Retrieval score returned by the underlying vector store.

        The interpretation depends on the vector database
        and similarity metric being used.

        Examples
        --------
        - Cosine distance
        - Cosine similarity
        - Dot product
        - Euclidean distance
        """

    chunk: Chunk
    score: float