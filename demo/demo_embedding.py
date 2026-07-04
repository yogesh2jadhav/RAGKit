from ragkit.chunkers.character_chunker import CharacterChunker
from ragkit.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from ragkit.loaders.loader_factory import LoaderFactory
from ragkit.sources.local_source import LocalSource


source = LocalSource("docs")

factory = LoaderFactory()

chunker = CharacterChunker()

embedder = SentenceTransformerEmbedder()

source_doc = next(source.discover())

loader = factory.get_loader(source_doc)

document = loader.load(source_doc)

chunks = list(
    chunker.chunk(document)
)

embeddings = list(
    embedder.embed(chunks)
)

print(
    len(embeddings[0].vector)
)