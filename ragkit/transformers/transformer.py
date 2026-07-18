"""
Purpose
-------
Defines the contract for transforming loaded documents.

Responsibilities
----------------
- Accept a Document.
- Return a transformed Document.

Does NOT
--------
- Load documents.
- Chunk documents.
- Generate embeddings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragkit.models.document import Document


class Transformer(ABC):
    """
    Base class for document transformers.
    """

    @abstractmethod
    def transform(
        self,
        document: Document,
    ) -> Document:
        """
        Transform a document before chunking.
        """
        raise NotImplementedError