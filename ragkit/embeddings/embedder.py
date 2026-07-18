"""
Purpose
-------
Defines the contract for generating embeddings from Chunks.

Responsibilities
----------------
- Accept Chunk objects.
- Produce Embedding objects.

Does NOT
--------
- Store embeddings.
- Perform similarity search.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from ragkit.models.chunk import Chunk
from ragkit.models.embedding import Embedding
from ragkit.models.query_embedding import QueryEmbedding

'''
=> Embedder is here is just like java Interface with one empty method some class
have to implement this.
'''
class Embedder(ABC):

    @abstractmethod
    def embed(
        self,
        chunks: Iterable[Chunk],
    ) -> Iterable[Embedding]:
        """
        Generate embeddings for the supplied chunks.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_query(
        self,
        query: str,
    ) -> QueryEmbedding:
        """
        Generate an embedding for a user query.
        """
        raise NotImplementedError
