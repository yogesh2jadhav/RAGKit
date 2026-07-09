"""
Purpose
-------
Hybrid retriever combining semantic and keyword search.

Responsibilities
----------------
- Retrieve documents using an existing Retriever.
- Perform keyword search.
- Merge both result sets.
- Remove duplicate chunks.

Does NOT
--------
- Generate embeddings.
- Perform vector search directly.
- Build prompts.
- Call an LLM.
"""

from __future__ import annotations

from ragkit.keyword.keyword_searcher import KeywordSearcher
from ragkit.models.search_result import SearchResult
from ragkit.retrievers.retriever import Retriever


class HybridRetriever(Retriever):
    """
    Hybrid semantic + keyword retriever.
    """

    def __init__(
        self,
        *,
        retriever: Retriever,
        keyword_searcher: KeywordSearcher,
    ) -> None:
        """
        Initialize the hybrid retriever.

        Parameters
        ----------
        retriever
            Retriever used for semantic retrieval.

        keyword_searcher
            Keyword search implementation.
        """

        self._retriever = retriever
        self._keyword_searcher = keyword_searcher

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Retrieve documents using hybrid retrieval.
        """

        vector_results = list(
            self._retriever.retrieve(
                query=query,
                top_k=top_k,
            )
        )

        keyword_results = list(
            self._keyword_searcher.search(
                query=query,
                top_k=top_k,
            )
        )

        merged: list[SearchResult] = []

        seen: set = set()

        for result in vector_results + keyword_results:

            if result.chunk.id in seen:
                continue

            seen.add(result.chunk.id)

            merged.append(result)

            if len(merged) >= top_k:
                break

        return merged