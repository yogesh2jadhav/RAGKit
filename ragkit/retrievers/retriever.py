"""
Purpose
-------
Defines the contract for retrieving relevant chunks.

Responsibilities
----------------
- Accept a user question.
- Return relevant chunks.

Does NOT
--------
- Generate LLM responses.
- Store vectors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragkit.models.chunk import Chunk

# This is an interface some class have to implement this.
class Retriever(ABC):

    @abstractmethod
    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[Chunk]:
        """
        Retrieve the most relevant chunks.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[Chunk]:
        """
        Perform similarity search.
        """
        raise NotImplementedError