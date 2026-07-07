"""
Example 03
----------

Generate embeddings for document chunks.

Concepts
--------
- Embedder
- Embedding
- Embedding Model
- Embedding Dimensions

Run
---
python examples/03_generate_embeddings.py

Previous
--------
02_chunk_documents.py

Next
----
04_build_vector_index.py
"""

from pathlib import Path

from ragkit.chunkers.character_chunker import CharacterChunker
from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.loaders.loader_factory import LoaderFactory
from ragkit.sources.local_source import LocalSource


def main() -> None:
    """
    Demonstrate embedding generation.
    """
    EXAMPLES_DIR = Path(__file__).parent
    DOCS_DIR = EXAMPLES_DIR / "docs"
    DATA_DIR = EXAMPLES_DIR / "data"
    VECTOR_DB_DIR = DATA_DIR / "vector_db"
    print("=" * 70)
    print("Example 03 - Generate Embeddings")
    print("=" * 70)

    docs_directory = DOCS_DIR

    if not docs_directory.exists():
        raise FileNotFoundError(
            "'docs' directory not found."
        )

    source = LocalSource(
        docs_directory,
    )

    chunker = CharacterChunker(
        chunk_size=300,
        chunk_overlap=50,
    )

    embedder = OllamaEmbedder(
        model="nomic-embed-text",
    )

    total_documents = 0
    total_chunks = 0
    total_embeddings = 0

    #
    # Process every discovered document.
    #
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

        embeddings = list(
            embedder.embed(
                chunks,
            )
        )

        total_chunks += len(chunks)
        total_embeddings += len(embeddings)

        print()
        print("=" * 70)
        print(f"Document : {source_document.uri}")
        print("=" * 70)

        print(f"Characters : {len(document.content)}")
        print(f"Chunks     : {len(chunks)}")
        print(f"Embeddings : {len(embeddings)}")

        #
        # Display the first two embeddings.
        #
        for chunk, embedding in zip(
            chunks[:2],
            embeddings[:2],
            strict=False,
        ):

            print()
            print("=" * 70)
            print(f"Chunk #{chunk.index}")
            print("=" * 70)

            print("Text")
            print("-" * 70)

            preview = chunk.content

            if len(preview) > 200:
                preview = preview[:200] + "..."

            print(preview)

            print()

            print(f"Embedding Model : {embedding.model}")
            print(f"Dimensions      : {len(embedding.vector)}")

            print()

            print("First 10 Values")
            print("-" * 70)

            for value in embedding.vector[:10]:
                print(f"{value:.6f}")

            print("-" * 70)

    #
    # Summary
    #
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"Documents Processed : {total_documents}")
    print(f"Chunks Generated    : {total_chunks}")
    print(f"Embeddings Created  : {total_embeddings}")

    #
    # Learning recap.
    #
    print()
    print("=" * 70)
    print("What did we learn?")
    print("=" * 70)

    print("✓ Embeddings convert text into numerical vectors.")
    print("✓ Every chunk generates exactly one embedding.")
    print("✓ The embedding model determines the vector values.")
    print("✓ Individual numbers have little meaning by themselves.")
    print("✓ The entire vector represents the semantic meaning of the text.")

    #
    # Knowledge Check
    #
    print()
    print("=" * 70)
    print("Knowledge Check")
    print("=" * 70)

    print("Q1. Why do we convert text into vectors?")
    print()
    print("Answer:")
    print("Computers compare vectors much more efficiently than raw text.")
    print()

    print("Q2. Does one embedding value have a human-readable meaning?")
    print()
    print("Answer:")
    print("No. Meaning is distributed across the entire embedding vector.")
    print()

    print("Q3. Does every chunk produce an embedding?")
    print()
    print("Answer:")
    print("Yes. Each chunk is converted into one embedding vector.")
    print()

    print("=" * 70)
    print("Next Example")
    print("=" * 70)
    print("04_build_vector_index.py")


if __name__ == "__main__":
    main()