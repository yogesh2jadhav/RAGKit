"""
Purpose
-------
Demonstrates storing embeddings in ChromaDB.

Pipeline
--------
LocalSource
    ↓
Loader
    ↓
Document
    ↓
CharacterChunker
    ↓
OllamaEmbedder
    ↓
ChromaVectorStore
"""

from ragkit.chunkers.character_chunker import CharacterChunker
from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.loaders.loader_factory import LoaderFactory
from ragkit.sources.local_source import LocalSource
from ragkit.vectorstores.chroma_vector_store import (
    ChromaVectorStore,
)


def main() -> None:

    source = LocalSource("docs")

    loader_factory = LoaderFactory()

    chunker = CharacterChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    embedder = OllamaEmbedder()

    store = ChromaVectorStore()

    total_embeddings = 0

    for source_document in source.discover():

        loader = loader_factory.get_loader(
            source_document,
        )

        document = loader.load(
            source_document,
        )

        chunks = chunker.chunk(
            document,
        )

        embeddings = list(
            embedder.embed(chunks),
        )

        store.add(
            embeddings,
        )

        total_embeddings += len(embeddings)

    print("=" * 60)
    print(f"Generated Embeddings : {total_embeddings}")
    print(f"Stored Embeddings    : {store.count()}")
    print("=" * 60)


if __name__ == "__main__":
    main()