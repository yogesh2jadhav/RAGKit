from ragkit.chunkers.character_chunker import CharacterChunker
from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.loaders.loader_factory import LoaderFactory
from ragkit.sources.local_source import LocalSource
from ragkit.vectorstores.chroma_vector_store import (
    ChromaVectorStore,
)


def main():

    source = LocalSource("docs")

    loader_factory = LoaderFactory()

    chunker = CharacterChunker()

    embedder = OllamaEmbedder()

    vector_store = ChromaVectorStore()

    for source_document in source.discover():

        loader = loader_factory.get_loader(
            source_document
        )

        document = loader.load(
            source_document
        )

        chunks = chunker.chunk(document)

        embeddings = embedder.embed(chunks)

        vector_store.add(
            embeddings
        )

    print(
        "Embeddings stored successfully."
    )


if __name__ == "__main__":
    main()