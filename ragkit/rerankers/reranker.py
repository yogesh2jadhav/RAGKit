"""
Purpose
-------
Defines the contract for reranking retrieved search results.

Responsibilities
----------------
- Accept retrieved search results.
- Return reordered search results.

Does NOT
--------
- Retrieve documents.
- Generate embeddings.
- Generate LLM responses.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from collections.abc import Iterable

from ragkit.models.search_result import SearchResult


class Reranker(ABC):
    """
    Base class for rerankers.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: Iterable[SearchResult],
    ) -> list[SearchResult]:
        """
        Reorder retrieved search results.
        """
        raise NotImplementedError