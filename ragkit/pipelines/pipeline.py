"""
Purpose
-------
Defines the contract for executable pipelines.

Responsibilities
----------------
- Accept a user query.
- Return an LLM response.

Does NOT
--------
- Retrieve documents directly.
- Generate embeddings directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragkit.models.llm_response import LLMResponse

# It is just like java interface. Some class have to build it.
class Pipeline(ABC):
    """
    Abstract base class for executable pipelines.
    """

    @abstractmethod
    def run(
        self,
        query: str,
    ) -> LLMResponse:
        """
        Execute the pipeline.

        Parameters
        ----------
        query
            User supplied query.

        Returns
        -------
        LLMResponse
        """
        raise NotImplementedError