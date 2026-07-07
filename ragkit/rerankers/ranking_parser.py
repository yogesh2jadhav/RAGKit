"""
Purpose
-------
Parses ranking responses produced by an LLM.

Responsibilities
----------------
- Parse ranking text.
- Return zero-based indices.

Does NOT
--------
- Call an LLM.
- Build prompts.
"""

from __future__ import annotations


class RankingParser:
    """
    Parses LLM ranking responses.
    """

    def parse(
        self,
        response: str,
    ) -> list[int]:
        """
        Parse a ranking response.

        Example
        -------
        2,1,3

        becomes

        [1,0,2]
        """

        indices: list[int] = []

        for value in response.split(","):

            value = value.strip()

            if not value:
                continue

            if not value.isdigit():
                continue

            indices.append(
                int(value) - 1,
            )

        return indices