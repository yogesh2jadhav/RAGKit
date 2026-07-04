"""
Purpose
-------
Stores embeddings inside a local ChromaDB database.

Responsibilities
----------------
- Create/Open a Chroma collection.
- Persist embeddings.
- Hide Chroma implementation details.

Does NOT
--------
- Generate embeddings.
- Perform LLM inference.
"""

from __future__ import annotations

from collections.abc import Iterable

import chromadb

from ragkit.models.embedding import Embedding
from ragkit.vectorstores.vector_store import VectorStore

'''
 There we are implemented VectorStore interface.
'''
class ChromaVectorStore(VectorStore):

    def __init__(
        self,
        path: str = "vector_db",
        collection_name: str = "ragkit",
    ) -> None:
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name
        )

    def add(
        self,
        embeddings: Iterable[Embedding],
    ) -> None:
        for embedding in embeddings:
            self._collection.add(  # _collection is connect to chromaDb and it will save id, vertor and metadata in db
                ids=[str(embedding.chunk_id)],
                documents=[embedding.content],
                embeddings=[embedding.vector],
                metadatas=[{"model": embedding.model}],
            )