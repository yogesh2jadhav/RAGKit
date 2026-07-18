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
from typing import Any

from ragkit.config.retrieval_config import RetrievalConfig
from ragkit.embeddings.embedder import Embedder
from ragkit.models.search_result import SearchResult
from ragkit.retrievers.retriever import Retriever
from ragkit.vectorstores.vector_store import VectorStore

'''
=> This class implement Retriever interface with one method called retrieve.
'''
class SimilarityRetriever(Retriever):
    """
    Retriever that performs vector similarity search.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        config: RetrievalConfig | None = None,
    ) -> None:
        """
        Initialize the retriever.
        """

        self._embedder = embedder
        self._vector_store = vector_store

        if config is None:
            config = RetrievalConfig()

        self._config = config

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> Iterable[SearchResult]:
        """
        Retrieve the most relevant search results.
        """

        #
        # Use the configured default unless
        # overridden for this request.
        #
        if top_k is None:
            top_k = self._config.top_k

        #
        # Convert the user's query into an embedding.
        #
        #=> Converting String query into embedding. for Verctor search
        query_embedding = self._embedder.embed_query(
            query,
        )

        #
        # Delegate similarity search to the vector store.
        #
        # => following is the code to make call to do query on vector db using vectorStore class

        return self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )
