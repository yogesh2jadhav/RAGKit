"""
Purpose
-------
Defines the contract for keyword-based search.

Responsibilities
----------------
- Accept a text query.
- Return ranked search results.

Does NOT
--------
- Generate embeddings.
- Perform vector similarity search.
- Generate LLM responses.
"""
# => This is just like java Interface with one empty method called search.
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable

from ragkit.models.search_result import SearchResult


class KeywordSearcher(ABC):
    """
    Base class for keyword search implementations.
    """

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> Iterable[SearchResult]:
        """
        Search for documents using keyword matching.
        """
        raise NotImplementedError