"""
Purpose
-------
Represents the outcome of an indexing operation.

Responsibilities
----------------
- Record indexing statistics.
- Provide a stable result model for indexers.

Does NOT
--------
- Perform indexing.
- Store documents.
- Generate embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IndexingResult:
    """
    Statistics produced by an indexing operation.

    Attributes
    ----------
    documents
        Number of indexed documents.

    chunks
        Number of generated chunks.

    embeddings
        Number of generated embeddings.
    """

    documents: int
    chunks: int
    embeddings: int