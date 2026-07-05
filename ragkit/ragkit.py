"""
Purpose
-------
Provides the main public API for RAGKit.

Responsibilities
----------------
- Coordinate document indexing.
- Coordinate query execution.

Does NOT
--------
- Build prompts.
- Generate embeddings.
- Retrieve documents.
- Generate LLM responses directly.
"""

from __future__ import annotations

from ragkit.indexers.indexer import Indexer
from ragkit.models.indexing_result import IndexingResult
from ragkit.models.llm_response import LLMResponse
from ragkit.pipelines.pipeline import Pipeline
from ragkit.sources.source import Source


class RagKit:
    """
    Main facade for RAGKit.
    """

    def __init__(
        self,
        *,
        indexer: Indexer,
        pipeline: Pipeline,
    ) -> None:
        self._indexer = indexer
        self._pipeline = pipeline

    def index(
        self,
        source: Source,
    ) -> IndexingResult:
        """
        Index all documents from a source.
        """

        return self._indexer.index(
            source,
        )

    def query(
        self,
        query: str,
    ) -> LLMResponse:
        """
        Execute a Retrieval-Augmented query.
        """

        return self._pipeline.invoke(
            query=query,
        )