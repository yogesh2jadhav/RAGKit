"""
Purpose
-------
Configuration for embedding models.

Responsibilities
----------------
- Store embedding configuration.

Does NOT
--------
- Generate embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

# => This is just like DTO in java


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """
    Configuration for an embedding model.
    """

    model: str = "nomic-embed-text"
