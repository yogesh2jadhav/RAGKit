from uuid import uuid4

from ragkit.chunkers.character_chunker import CharacterChunker
from ragkit.models.document import Document


def test_character_chunker():

    document = Document(
        id=uuid4(),
        content="A" * 2500,
    )

    chunker = CharacterChunker(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = list(chunker.chunk(document))

    assert len(chunks) == 4

    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == 1000

    assert chunks[1].start_offset == 800