"""
Purpose
-------
Stores embeddings inside a local Chroma database.

Responsibilities
----------------
- Create/Open a Chroma collection.
- Persist embeddings.
- Persist complete Chunk metadata.

Does NOT
--------
- Generate embeddings.
- Perform similarity search.
"""

from __future__ import annotations

from collections.abc import Iterable

import chromadb

from ragkit.models.chunk import Chunk
from ragkit.models.embedding import Embedding
from ragkit.vectorstores.vector_store import VectorStore


class ChromaVectorStore(VectorStore):
    """
    ChromaDB implementation of VectorStore.
    """

    def __init__(
        self,
        path: str = "./vector_db",
        collection_name: str = "ragkit",
    ) -> None:

        self._client = chromadb.PersistentClient(
            path=path,
        )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
        )

    def add(
        self,
        chunks: Iterable[Chunk],
        embeddings: Iterable[Embedding],
    ) -> None:
        """
        Store chunks together with their embeddings.

        Each Chunk must correspond to exactly one Embedding.

        The Chunk remains the source of truth for all metadata.
        """

        #
        # Iterate over both collections together.
        #
        # zip(..., strict=True) ensures:
        #
        #   1. Same number of chunks and embeddings.
        #   2. Raises ValueError if they differ.
        #
        for chunk, embedding in zip(chunks, embeddings, strict=True):

            #
            # Defensive validation.
            #
            # This catches programming mistakes immediately instead of
            # silently corrupting the vector database.
            #
            if chunk.id != embedding.chunk_id:
                raise ValueError(
                    "Chunk.id does not match Embedding.chunk_id."
                )

            #
            # Store complete metadata.
            #
            metadata = {
                "model": embedding.model,
                "document_id": str(chunk.document_id),
                "chunk_index": chunk.index,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
                **chunk.metadata,
            }

            self._collection.add(
                ids=[str(chunk.id)],
                embeddings=[embedding.vector],
                documents=[chunk.content],
                metadatas=[metadata],
            )

    def count(self) -> int:
        """
        Returns the number of stored vectors.
        """
        return self._collection.count()