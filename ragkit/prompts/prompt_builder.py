"""
Purpose
-------
Defines the contract for building prompts for an LLM.

Responsibilities
----------------
- Accept a user query.
- Accept retrieved search results.
- Produce a prompt for an LLM.

Does NOT
--------
- Retrieve documents.
- Generate embeddings.
- Call an LLM.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from ragkit.models.search_result import SearchResult


#=> This is just like interface with build as a method in java
class PromptBuilder(ABC):
    """
    Abstract base class for all prompt builders.
    """

    @abstractmethod
    def build(
        self,
        query: str,
        search_results: Iterable[SearchResult],
    ) -> str:
        """
        Build an LLM prompt.

        Parameters
        ----------
        query
            User supplied query.

        search_results
            Retrieved search results.

        Returns
        -------
        str
            Prompt ready for the LLM.
        """
        raise NotImplementedError