"""
Purpose
-------
Default reranker that performs no reranking.

Responsibilities
----------------
- Preserve the original ranking.

Does NOT
--------
- Score documents.
- Change document ordering.
"""

from __future__ import annotations

from collections.abc import Iterable

from ragkit.models.search_result import SearchResult
from ragkit.rerankers.reranker import Reranker


class IdentityReranker(Reranker):
    """
    Returns search results unchanged.
    """

    def rerank(
        self,
        query: str,
        results: Iterable[SearchResult],
    ) -> list[SearchResult]:

        return list(results)