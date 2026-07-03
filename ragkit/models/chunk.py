"""
Purpose
-------
Represents a retrievable portion of a document.

Responsibilities
----------------
- Hold a section of document content.
- Keep the relationship to the parent document.
- Preserve chunk ordering.
- Store chunk metadata.

Does NOT
--------
- Store embeddings.
- Generate embeddings.
- Know how it was created.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Chunk:
    """
    A retrievable unit derived from a Document.
    """

    id: UUID
    document_id: UUID
    index: int
    content: str
    start_offset: int
    end_offset: int
    metadata: dict[str, Any] = field(default_factory=dict)
