"""
Purpose
-------
Defines the contract for indexing documents into a vector store.

Responsibilities
----------------
- Accept a Source.
- Index documents into a VectorStore.

Does NOT
--------
- Answer user queries.
- Generate LLM responses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragkit.sources.source import Source


class Indexer(ABC):
    """
    Abstract base class for all document indexers.
    """

    @abstractmethod
    def index(
        self,
        source: Source,
    ) -> None:
        """
        Index all documents from the supplied source.
        """
        raise NotImplementedError