"""
Purpose
-------
Demonstrates embedding generation using Ollama.

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
"""

from ragkit.chunkers.character_chunker import CharacterChunker
from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.loaders.loader_factory import LoaderFactory
from ragkit.sources.local_source import LocalSource


def main() -> None:

    source = LocalSource("docs")

    chunker = CharacterChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    embedder = OllamaEmbedder()

    for source_document in source.discover():

        loader = LoaderFactory.get_loader(
            source_document
        )

        document = loader.load(
            source_document
        )

        chunks = chunker.chunk(
            document
        )

        print("=" * 80)
        print(document.metadata["filename"])
        print("=" * 80)

        for embedding in embedder.embed(
            chunks
        ):

            print(
                f"Chunk : {embedding.chunk_id}"
            )

            print(
                f"Model : {embedding.model}"
            )

            print(
                f"Vector Dimension : {len(embedding.vector)}"
            )

            print()

            #
            # Show only first embedding
            #
            break


if __name__ == "__main__":
    main()