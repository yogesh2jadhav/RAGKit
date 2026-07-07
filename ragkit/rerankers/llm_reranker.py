"""
Purpose
-------
Reranks retrieved search results using
a Large Language Model.

Responsibilities
----------------
- Build a ranking request.
- Ask an LLM to rank results.
- Return reordered search results.

Does NOT
--------
- Retrieve documents.
- Generate embeddings.
"""

from __future__ import annotations

from collections.abc import Iterable

from ragkit.config.reranker_config import RerankerConfig
from ragkit.llms.llm import LLM
from ragkit.models.search_result import SearchResult
from ragkit.rerankers.reranker import Reranker


class LLMReranker(Reranker):
    """
    Reranker powered by an LLM.
    """

    def __init__(
        self,
        *,
        llm: LLM,
        config: RerankerConfig | None = None,
    ) -> None:

        self._llm = llm

        if config is None:
            config = RerankerConfig()

        self._config = config

    def rerank(
        self,
        query: str,
        results: Iterable[SearchResult],
    ) -> list[SearchResult]:
        """
        Rerank retrieved search results.

        This implementation will be completed
        in the next commit.
        """

        return list(results)