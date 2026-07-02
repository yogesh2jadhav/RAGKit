"""
Purpose
-------
Splits a document into overlapping character-based chunks.

Responsibilities
----------------
- Split a document into chunks.
- Preserve chunk order.
- Apply overlap.

Does NOT
--------
- Generate embeddings.
- Store chunks.
- Retrieve chunks.

Notes
-----
This is Version 1.

Future versions will support:
- Sentence chunking
- Paragraph chunking
- Token-aware chunking
- Recursive chunking
"""

from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

from ragkit.chunkers.chunker import Chunker
from ragkit.models.chunk import Chunk
from ragkit.models.document import Document


class RecursiveChunker(Chunker):

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
        '''
         content will have all document data as string in it.
        '''
        content = document.content

        start = 0
        index = 0

        while start < len(content):
            end = start + self._chunk_size
            chunk_text = content[start:end] # Here we are just doing string split.
            yield Chunk(    # yield = we use for Streaming.
                id=uuid4(),
                document_id=document.id,
                index=index,
                content=chunk_text,
                metadata=document.metadata.copy(),
            )

            index += 1

            start += (
                self._chunk_size
                - self._chunk_overlap
            )