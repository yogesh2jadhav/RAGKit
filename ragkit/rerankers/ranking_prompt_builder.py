"""
Purpose
-------
Builds prompts for LLM-based reranking.

Responsibilities
----------------
- Convert a query and search results into
  a ranking prompt.

Does NOT
--------
- Call the LLM.
- Parse LLM responses.
"""

from __future__ import annotations

from collections.abc import Iterable

from ragkit.models.search_result import SearchResult


class RankingPromptBuilder:
    """
    Builds prompts for LLM reranking.
    """

    def build(
        self,
        *,
        query: str,
        search_results: Iterable[SearchResult],
    ) -> str:

        lines = [
            "You are a search ranking assistant.",
            "",
            "Rank the following passages by relevance.",
            "",
            f"Query: {query}",
            "",
            "Return ONLY the passage numbers.",
            "Example:",
            "2,1,3",
            "",
            "Passages:",
            "",
        ]

        for index, result in enumerate(
            search_results,
            start=1,
        ):
            lines.append(f"{index}.")
            lines.append(result.chunk.content)
            lines.append("")

        return "\n".join(lines)