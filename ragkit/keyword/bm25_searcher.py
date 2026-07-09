"""
Purpose
-------
BM25 keyword search implementation.

Responsibilities
----------------
- Build a BM25 index.
- Perform lexical search.
- Return ranked search results.

Does NOT
--------
- Generate embeddings.
- Perform vector search.
- Call an LLM.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from rank_bm25 import BM25Okapi

from ragkit.keyword.keyword_searcher import KeywordSearcher
from ragkit.models.chunk import Chunk
from ragkit.models.search_result import SearchResult
from ragkit.vectorstores.vector_store import VectorStore


class BM25Searcher(KeywordSearcher):
    """
    BM25 keyword search implementation.
    """

    def __init__(
        self,
        *,
        vector_store: VectorStore,
    ) -> None:

        self._vector_store = vector_store

        self._chunks: list[Chunk] = []

        self._bm25: BM25Okapi | None = None

        self._build_index()

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> Iterable[SearchResult]:
        """
        Perform BM25 keyword search.
        """

        if self._bm25 is None:
            raise RuntimeError("BM25 index has not been built.")

        query_tokens = self._tokenize(
            query,
        )

        scores = self._bm25.get_scores(
            query_tokens,
        )

        ranked = sorted(
            zip(
                self._chunks,
                scores,
                strict=True,
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        for chunk, score in ranked[:top_k]:

            yield SearchResult(
                chunk=chunk,
                score=float(score),
            )

    def _build_index(
        self,
    ) -> None:
        """
        Build the BM25 index.
        """

        chunks = list(self._vector_store.iter_chunks())

        self._chunks = chunks

        corpus = [
            self._tokenize(
                chunk.content,
            )
            for chunk in self._chunks
        ]

        self._bm25 = BM25Okapi(
            corpus,
        )

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:
        """
        Tokenize text.
        """

        return re.findall(
            r"\w+",
            text.lower(),
        )