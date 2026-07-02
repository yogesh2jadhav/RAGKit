"""
Purpose
-------
Defines the contract for converting a Document into one or more Chunks.

Responsibilities
----------------
- Accept a Document.
- Produce Chunk objects.

Does NOT
--------
- Generate embeddings.
- Store chunks.
- Retrieve chunks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from ragkit.models.chunk import Chunk
from ragkit.models.document import Document


class Chunker(ABC):

    @abstractmethod
    def chunk(
        self,
        document: Document,
    ) -> Iterable[Chunk]:
        """
        Split a document into chunks.
        """
        raise NotImplementedError