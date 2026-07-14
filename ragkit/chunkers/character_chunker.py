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
from ragkit.config.chunker_config import ChunkerConfig

'''
=> CharacterChunker is the class which have implement Chunker interface.
'''
class CharacterChunker(Chunker):

    '''
     => following is class constructor which will take chunkSize and OverLapping size as input
        if all ok it will set both chunkSize and chunk_overlap.
    '''
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        config: ChunkerConfig | None = None
    ) -> None:

        if config is not None:
            chunk_size = config.chunk_size

        if config is not None:
            chunk_overlap = config.chunk_overlap

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    '''
    => this method will break document into chunk in defined size. 
       This method act like streamer, 
       it will create one chunk and yield that one chunks (return one chunk) and loop continue for next chunk.
    '''
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

            yield Chunk(
                id=uuid4(),
                document_id=document.id,
                index=index,
                start_offset=start,
                end_offset=end,
                content=content[start:end],
                metadata=dict(document.metadata),
            )

            start += (
                self._chunk_size
                - self._chunk_overlap
            )

            index += 1