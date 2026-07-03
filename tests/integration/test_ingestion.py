from ragkit.chunkers.character_chunker import CharacterChunker
from ragkit.loaders.loader_factory import LoaderFactory
from ragkit.sources.local_source import LocalSource


def test_ingestion_pipeline():

    source = LocalSource("docs")

    factory = LoaderFactory()

    chunker = CharacterChunker()

    source_document = next(source.discover())

    loader = factory.get_loader(source_document)

    document = loader.load(source_document)

    chunks = list(chunker.chunk(document))

    assert len(chunks) > 0