"""
Purpose
-------
Defines the contract for retrieving relevant content.

Responsibilities
----------------
- Accept a search query.
- Return ranked search results.

Does NOT
--------
- Generate embeddings.
- Perform vector search directly.
- Generate LLM responses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from ragkit.models.search_result import SearchResult

#=> This is just like interface with one method
class Retriever(ABC):
    """
    Abstract base class for all retrieval strategies.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> Iterable[SearchResult]:
        """
        Retrieve the most relevant search results.

        Parameters
        ----------
        query
            User supplied search query.

        top_k
            Maximum number of results to retrieve.

        Returns
        -------
        Iterable[SearchResult]
            Search results ordered by relevance.
        """
        raise NotImplementedError