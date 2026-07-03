"""
Purpose
-------
Splits a document into fixed-size overlapping character chunks.

Responsibilities
----------------
- Split document into chunks.
- Apply overlap.
- Preserve ordering.

Does NOT
--------
- Understand sentences.
- Understand paragraphs.
- Generate embeddings.
"""

from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

from ragkit.chunkers.chunker import Chunker
from ragkit.models.chunk import Chunk
from ragkit.models.document import Document


class CharacterChunker(Chunker):

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(
        self,
        document: Document,
    ) -> Iterable[Chunk]:
        """
        content will have all document data as string in it.
        """
        content = document.content

        start = 0
        index = 0

        while start < len(content):

            end = min(
                start + self._chunk_size,
                len(content),
            )

            yield Chunk( # yield = we use for Streaming.
                id=uuid4(),
                document_id=document.id,
                index=index,
                start_offset=start,
                end_offset=end,
                content=content[start:end],# Here we are just doing string split.
            )

            start += (
                self._chunk_size
                - self._chunk_overlap
            )

            index += 1