"""
Purpose
-------
Represents an embedding generated from a user's search query.

Responsibilities
----------------
- Hold the original query text.
- Hold the embedding model name.
- Hold the embedding vector.
- Be immutable after creation.

Does NOT
---------
- Represent document chunks.
- Store document metadata.
- Perform embedding generation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QueryEmbedding:
    """
    Represents an embedded user query.

    Attributes
    ----------

    model:
        Embedding model used to generate the vector.

    vector:
        Numerical embedding vector.
    """
    model: str
    vector: list[float]