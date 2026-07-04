"""
Purpose
-------
Retrieves the most relevant chunks using vector similarity search.

Responsibilities
----------------
- Convert a query into an embedding.
- Delegate similarity search to the VectorStore.
- Return search results.

Does NOT
--------
- Generate LLM responses.
- Store vectors.
- Rank or rerank results.
"""

from __future__ import annotations

from collections.abc import Iterable

from ragkit.embeddings.embedder import Embedder
from ragkit.models.search_result import SearchResult
from ragkit.vectorstores.vector_store import VectorStore

from ragkit.retrievers.retriever import Retriever


class SimilarityRetriever(Retriever):
    """
    Retriever that performs vector similarity search.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> Iterable[SearchResult]:
        """
        Retrieve the most relevant search results.
        """

        query_embedding = self._embedder.embed_query(
            query,
        )

        return self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )