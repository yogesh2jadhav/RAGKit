from ragkit.chunkers.character_chunker import CharacterChunker
from ragkit.loaders.loader_factory import LoaderFactory
from ragkit.sources.local_source import LocalSource


def main():

    source = LocalSource("docs")

    factory = LoaderFactory()

    chunker = CharacterChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    for source_document in source.discover():

        loader = factory.get_loader(source_document)

        document = loader.load(source_document)

        print("=" * 80)
        print(document.metadata["filename"])
        print("=" * 80)

        for chunk in chunker.chunk(document):

            print(
                f"Chunk {chunk.index}"
            )

            print(chunk.content)

            print("-" * 80)


if __name__ == "__main__":
    main()