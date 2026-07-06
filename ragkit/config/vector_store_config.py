"""
Purpose
-------
Configuration for vector stores.
"""

from __future__ import annotations

from dataclasses import dataclass

# => This is just like DTO in java

@dataclass(frozen=True, slots=True)
class VectorStoreConfig:
    """
    Configuration for a vector store.
    """

    path: str = "./vector_db"
    collection_name: str = "ragkit"