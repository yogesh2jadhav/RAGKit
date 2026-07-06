"""
Purpose
-------
Indexes documents from a Source into a VectorStore.

Responsibilities
----------------
- Discover source documents.
- Load documents.
- Chunk documents.
- Generate embeddings.
- Store embeddings in the VectorStore.
- Return indexing statistics.

Does NOT
--------
- Perform similarity search.
- Generate LLM responses.
"""

from __future__ import annotations

from ragkit.chunkers.chunker import Chunker
from ragkit.embeddings.embedder import Embedder
from ragkit.indexers.indexer import Indexer
from ragkit.loaders.loader_factory import LoaderFactory
from ragkit.models.indexing_result import IndexingResult
from ragkit.sources.source import Source
from ragkit.vectorstores.vector_store import VectorStore
from ragkit.transformers.identity_transformer import IdentityTransformer
from ragkit.transformers.transformer import Transformer

class DocumentIndexer(Indexer):
    """
    Default implementation of Indexer.
    => Thie class shows load directory -> Read document -> create chunk -> do embeddings -> save to vector store.
    """

    def __init__(
        self,
        *,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        transformer: Transformer | None = None,
    ) -> None:
        """
        Parameters
        ----------
        chunker Splits documents into chunks.
        embedder Generates embeddings for chunks.
        vector_store Stores generated embeddings.
        transformer Transforms documents before chunking.
        """

        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store

        if transformer is None:
            transformer = IdentityTransformer()

        self._transformer = transformer

    def index(
        self,
        source: Source,
    ) -> IndexingResult:
        """
        Index every document discovered from the supplied source.
        """

        document_count = 0
        chunk_count = 0
        embedding_count = 0

        #
        # Process one document at a time.
        #
        # This keeps memory usage nearly constant,
        # regardless of the total number of documents.
        #
        for source_document in source.discover():

            loader = LoaderFactory.get_loader(
                source_document,
            )

            document = loader.load(
                source_document,
            )
            #
            # Transform the document before chunking.
            #
            document = self._transformer.transform(
                document,
            )
            document_count += 1

            #
            # Materialize the chunks because they are
            # consumed twice:
            #
            #   1. Embedder
            #   2. VectorStore
            #
            chunks = list(
                self._chunker.chunk(
                    document,
                )
            )

            chunk_count += len(chunks)

            embeddings = list(
                self._embedder.embed(
                    chunks,
                )
            )

            embedding_count += len(embeddings)

            self._vector_store.add(
                chunks=chunks,
                embeddings=embeddings,
            )

        return IndexingResult(
            documents=document_count,
            chunks=chunk_count,
            embeddings=embedding_count,
        )