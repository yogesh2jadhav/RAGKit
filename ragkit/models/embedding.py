"""
Purpose
-------
Represents the vector embedding of a Chunk.

Responsibilities
----------------
- Store the generated embedding vector.
- Maintain the relationship to the Chunk.
- Record which embedding model generated the vector.

Does NOT
--------
- Generate embeddings.
- Store chunks.
- Interact with a vector database.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

'''
 Embedding is just DTO in java
'''
@dataclass(frozen=True, slots=True)
class Embedding:
    chunk_id: UUID
    content: str
    model: str
    vector: list[float]