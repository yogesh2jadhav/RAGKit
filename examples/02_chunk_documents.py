"""
Example 02
----------

Split a document into chunks.

Concepts
--------
- Chunker
- Chunk
- Chunk Size
- Chunk Overlap

Run
---
python examples/02_chunk_documents.py

Previous
--------
01_load_documents.py

Next
----
03_generate_embeddings.py
"""

from pathlib import Path

from ragkit.chunkers.character_chunker import CharacterChunker
from ragkit.loaders.loader_factory import LoaderFactory
from ragkit.sources.local_source import LocalSource


def main() -> None:
    """
    Demonstrate document chunking.
    """
    EXAMPLES_DIR = Path(__file__).parent
    DOCS_DIR = EXAMPLES_DIR / "docs"
    print("=" * 70)
    print("Example 02 - Chunk Documents")
    print("=" * 70)

    docs_directory = Path(__file__).parent / "docs"

    if not docs_directory.exists():
        raise FileNotFoundError(
            "'docs' directory not found."
        )

    #
    # Discover documents.
    #
    source = LocalSource(
        DOCS_DIR,
    )

    #
    # Character based chunker.
    #
    chunker = CharacterChunker(
        chunk_size=300,
        chunk_overlap=50,
    )

    total_documents = 0
    total_chunks = 0

    for source_document in source.discover():

        loader = LoaderFactory.get_loader(
            source_document,
        )

        document = loader.load(
            source_document,
        )

        total_documents += 1

        chunks = list(
            chunker.chunk(
                document,
            )
        )

        total_chunks += len(chunks)

        print()
        print("=" * 70)
        print(f"Document : {source_document.uri}")
        print("=" * 70)

        print(f"Characters : {len(document.content)}")
        print(f"Chunks     : {len(chunks)}")

        #
        # Display the first few chunks.
        #
        for chunk in chunks[:3]:
            print()
            print("=" * 70)
            print(f"Chunk #{chunk.index}")
            print("=" * 70)

            print(f"Start Offset : {chunk.start_offset}")
            print(f"End Offset   : {chunk.end_offset}")
            print(f"Length       : {len(chunk.content)}")

            #
            # Show how much overlap exists with
            # the previous chunk.
            #
            if chunk.index > 0:
                overlap = chunks[chunk.index - 1].end_offset - chunk.start_offset

                print(f"Overlap      : {overlap}")

            print("-" * 70)

            preview = chunk.content

            if len(preview) > 250:
                preview = preview[:250] + "..."

            print(preview)

            print("-" * 70)

        if len(chunks) > 3:
            print()

            print(f"... {len(chunks) - 3} additional chunks omitted ...")

    #
    # Summary
    #
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"Documents Processed : {total_documents}")
    print(f"Total Chunks        : {total_chunks}")

    #
    # Learning recap.
    #
    print()
    print("=" * 70)
    print("What did we learn?")
    print("=" * 70)

    print("✓ A Document can be too large for efficient retrieval.")
    print("✓ A Chunker divides a document into smaller chunks.")
    print("✓ Chunk size controls how much information each chunk contains.")
    print("✓ Chunk overlap preserves context across chunk boundaries.")

    #
    # Reinforce the concepts.
    #
    print()
    print("=" * 70)
    print("Knowledge Check")
    print("=" * 70)

    print("Q1. Why don't we embed the entire document?")
    print()
    print("Answer:")
    print("Large documents reduce retrieval accuracy and may exceed")
    print("LLM context limits. Smaller chunks are easier to search.")
    print()

    print("Q2. Why do chunks overlap?")
    print()
    print("Answer:")
    print("Important information may span two chunk boundaries.")
    print("Overlap ensures that the surrounding context is preserved.")
    print()

    print("Q3. What happens if chunk_overlap = 0?")
    print()
    print("Answer:")
    print("Context can be split between adjacent chunks, reducing")
    print("retrieval quality for information near chunk boundaries.")
    print()

    print("=" * 70)
    print("Next Example")
    print("=" * 70)
    print("03_generate_embeddings.py")


if __name__ == "__main__":
    main()
