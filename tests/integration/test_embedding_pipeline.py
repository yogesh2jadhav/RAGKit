from unittest.mock import Mock

from ragkit.chunkers.character_chunker import CharacterChunker
from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.loaders.loader_factory import LoaderFactory
from ragkit.sources.local_source import LocalSource


def test_embedding_pipeline():

    source = LocalSource("docs")

    source_document = next(
        source.discover()
    )

    loader = LoaderFactory.get_loader(
        source_document
    )

    document = loader.load(
        source_document
    )

    chunker = CharacterChunker()

    chunks = list(
        chunker.chunk(document)
    )

    embedder = OllamaEmbedder()

    embedder._client.embed = Mock(
        side_effect=[
            Mock(
                embeddings=[
                    [0.1, 0.2, 0.3]
                ]
            )
            for _ in chunks
        ]
    )

    embeddings = list(
        embedder.embed(chunks)
    )

    assert len(embeddings) == len(chunks)

    assert len(
        embeddings[0].vector
    ) > 0
