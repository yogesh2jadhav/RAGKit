"""
Purpose
-------
Processes a loaded document.

Responsibilities
----------------
- Transform the document.
- Chunk the document.
- Generate embeddings.

Does NOT
--------
- Load documents.
- Store vectors.
- Discover sources.
"""

from __future__ import annotations

from ragkit.chunkers.chunker import Chunker
from ragkit.embeddings.embedder import Embedder
from ragkit.models.chunk import Chunk
from ragkit.models.document import Document
from ragkit.models.embedding import Embedding
from ragkit.transformers.identity_transformer import IdentityTransformer
from ragkit.transformers.transformer import Transformer


class DocumentProcessor:
    """
    Processes a single document.
    """

    def __init__(
        self,
        *,
        transformer: Transformer | None = None,
        chunker: Chunker,
        embedder: Embedder,
    ) -> None:

        self._transformer = transformer or IdentityTransformer()
        self._chunker = chunker
        self._embedder = embedder

    def process(
        self,
        document: Document,
    ) -> tuple[list[Chunk], list[Embedding]]:
        """
        Process a document into chunks and embeddings.
        """

        document = self._transformer.transform(
            document,
        )

        chunks = list(
            self._chunker.chunk(
                document,
            )
        )

        embeddings = list(
            self._embedder.embed(
                chunks,
            )
        )

        return chunks, embeddings