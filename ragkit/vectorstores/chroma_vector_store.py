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
from ragkit.models.query_embedding import QueryEmbedding
from ragkit.models.search_result import SearchResult
from uuid import UUID
class ChromaVectorStore(VectorStore):
    """
    ChromaDB implementation of VectorStore.
    """

    #
    # Metadata keys stored inside Chroma.
    #
    # Using constants prevents accidental typos and keeps
    # the storage and retrieval logic synchronized.
    #
    _MODEL = "model"
    _DOCUMENT_ID = "document_id"
    _CHUNK_INDEX = "chunk_index"
    _START_OFFSET = "start_offset"
    _END_OFFSET = "end_offset"
    
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

        embedding_by_chunk_id = {embedding.chunk_id: embedding for embedding in embeddings}

        for chunk in chunks:
            embedding = embedding_by_chunk_id.get(chunk.id)

            if embedding is None:
                raise ValueError(f"No embedding found for chunk {chunk.id}.")

            metadata = {
                self._MODEL: embedding.model,
                self._DOCUMENT_ID: str(chunk.document_id),
                self._CHUNK_INDEX: chunk.index,
                self._START_OFFSET: chunk.start_offset,
                self._END_OFFSET: chunk.end_offset,
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

    def search(
        self,
        query_embedding: QueryEmbedding,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Search for the most similar chunks.

        Responsibilities
        ----------------
        - This is currently a placeholder implementation.
        - It exists so the project remains buildable while the
          retrieval feature is being developed.

        Does NOT
        --------
        - Perform an actual similarity search.
        """

        raise NotImplementedError(
            "ChromaVectorStore.search() has not been implemented yet."
        )
    def _build_chunk(
        self,
        *,
        chunk_id: str,
        document: str,
        metadata: dict,
    ) -> Chunk:
        """
        Reconstruct a Chunk from data stored in Chroma.

        Responsibilities
        ----------------
        - Convert Chroma metadata back into a Chunk.
        - Restore UUID fields.
        - Preserve custom metadata.

        Does NOT
        --------
        - Query Chroma.
        - Perform similarity search.
        """

        #
        # Remove internal metadata fields.
        #
        # Everything remaining belongs to the original chunk.
        #
        custom_metadata = dict(metadata)

        custom_metadata.pop(self._MODEL, None)

        document_id = UUID(custom_metadata.pop(self._DOCUMENT_ID))

        index = custom_metadata.pop(self._CHUNK_INDEX)

        start_offset = custom_metadata.pop(self._START_OFFSET)

        end_offset = custom_metadata.pop(self._END_OFFSET)

        return Chunk(
            id=UUID(chunk_id),
            document_id=document_id,
            index=index,
            content=document,
            start_offset=start_offset,
            end_offset=end_offset,
            metadata=custom_metadata,
        )