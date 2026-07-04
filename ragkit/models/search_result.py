"""
Purpose
-------
Represents the result returned by a vector similarity search.

Responsibilities
----------------
- Hold the retrieved Chunk.
- Hold the similarity score for that Chunk.

Does NOT
--------
- Perform vector search.
- Rank results.
- Generate embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragkit.models.chunk import Chunk

# This is just like DTO in java
@dataclass(frozen=True, slots=True)
class SearchResult:
    """
    Represents a single search result returned by a VectorStore.

    Attributes
    ----------
    chunk
        The retrieved chunk.

    similarity
        Similarity score between the query embedding
        and the stored chunk embedding.

        Higher values indicate better matches.
    """

    chunk: Chunk
    similarity: float