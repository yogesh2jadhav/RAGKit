"""
Purpose
-------
Demonstrates the complete embedding pipeline.

Pipeline
--------
Local Folder
    ↓
Document Loader
    ↓
Character Chunker
    ↓
Ollama Embedder
"""

from ragkit.chunkers.character_chunker import CharacterChunker
from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.loaders.loader_factory import LoaderFactory
from ragkit.sources.local_source import LocalSource


def main() -> None:

    source = LocalSource("docs") # This will load all file list from directory.

    factory = LoaderFactory()

    chunker = CharacterChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    embedder = OllamaEmbedder()

    for source_document in source.discover():  # source.discover() is list and source_document will hold one file at time.

        loader = factory.get_loader(source_document)
        document = loader.load(source_document)
        chunks = chunker.chunk(document)
        embeddings = embedder.embed(chunks)
        print("=" * 80)
        print(document.metadata["filename"])
        print("=" * 80)

        for embedding in embeddings:

            print(
                f"Vector Length : {len(embedding.vector)}"
            )

            break


if __name__ == "__main__":
    main()