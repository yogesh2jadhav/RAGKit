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
from ragkit.rerankers.ranking_parser import RankingParser
from ragkit.rerankers.ranking_prompt_builder import RankingPromptBuilder
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

        self._prompt_builder = RankingPromptBuilder()

        self._parser = RankingParser()

        if config is None:
            config = RerankerConfig()

        self._config = config

    def rerank(
        self,
        query: str,
        results: Iterable[SearchResult],
    ) -> list[SearchResult]:
        """
        Rerank search results using an LLM.
        """

        #
        # Convert iterable into a list because
        # we'll iterate multiple times.
        #
        results = list(results)

        if not results:
            return []

        #
        # Build ranking prompt.
        #
        prompt = self._prompt_builder.build(
            query=query,
            search_results=results,
        )

        #
        # Ask the LLM to rank passages.
        #
        response = self._llm.generate(
            prompt=prompt,
            options={
                "temperature": self._config.temperature,
                "num_predict": self._config.max_tokens,
            },
        )

        #
        # Parse ranking.
        #
        ranking = self._parser.parse(
            response.content,
        )

        reranked: list[SearchResult] = []

        used: set[int] = set()

        #
        # Add ranked results.
        #
        for index in ranking:
            if index < 0:
                continue

            if index >= len(results):
                continue

            if index in used:
                continue

            reranked.append(
                results[index],
            )

            used.add(index)

        #
        # Preserve remaining results.
        #
        for index, result in enumerate(results):
            if index not in used:
                reranked.append(
                    result,
                )

        #
        # Return configured number of results.
        #
        return reranked[: self._config.top_k]