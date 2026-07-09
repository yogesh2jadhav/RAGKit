"""
Purpose
-------
Defines the interface for keyword-based search.

Responsibilities
----------------
- Perform lexical (keyword) search.
- Return ranked search results.

Does NOT
--------
- Generate embeddings.
- Perform vector search.
- Merge search results.
- Call an LLM.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable

from ragkit.models.search_result import SearchResult


class KeywordSearcher(ABC):
    """
    Abstract base class for keyword search.

    Concrete implementations may use BM25,
    Lucene, Elasticsearch, SQLite FTS,
    or any other keyword search algorithm.
    """

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> Iterable[SearchResult]:
        """
        Search for documents using keywords.

        Parameters
        ----------
        query
            User query.

        top_k
            Maximum number of search results.

        Returns
        -------
        Iterable[SearchResult]
            Ranked search results.
        """
        raise NotImplementedError