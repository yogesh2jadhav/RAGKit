"""
Purpose
-------
Defines the contract for storing and searching vector embeddings.

Responsibilities
----------------
- Store embeddings.
- Perform similarity search.

Does NOT
--------
- Generate embeddings.
- Chunk documents.
- Load files.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from ragkit.models.embedding import Embedding

'''
 This is VectorStore interface some class have to immplement this.
'''
class VectorStore(ABC):

    @abstractmethod
    def add(
        self,
        embeddings: Iterable[Embedding],
    ) -> None:
        """
        Store embeddings.
        """
        raise NotImplementedError