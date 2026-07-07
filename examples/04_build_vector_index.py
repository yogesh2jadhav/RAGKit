"""
Example 04
----------

Build a vector index from documents.

Concepts
--------
- Vector Store
- Persistent Index
- Indexing
- Embedding Storage

Run
---
python examples/04_build_vector_index.py

Previous
--------
03_generate_embeddings.py

Next
----
05_similarity_search.py
"""

from pathlib import Path

from ragkit.chunkers.character_chunker import CharacterChunker
from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.indexers.document_indexer import DocumentIndexer
from ragkit.processors.document_processor import DocumentProcessor
from ragkit.sources.local_source import LocalSource
from ragkit.transformers.identity_transformer import IdentityTransformer
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore


def main() -> None:
    """
    Build a persistent vector index.
    """
    EXAMPLES_DIR = Path(__file__).parent
    DOCS_DIR = EXAMPLES_DIR / "docs"
    DATA_DIR = EXAMPLES_DIR / "data"
    VECTOR_DB_DIR = DATA_DIR / "vector_db"

    print("=" * 70)
    print("Example 04 - Build Vector Index")
    print("=" * 70)

    docs_directory = DOCS_DIR

    if not docs_directory.exists():
        raise FileNotFoundError(
            "'docs' directory not found."
        )

    source = LocalSource(
        docs_directory,
    )

    processor = DocumentProcessor(
        transformer=IdentityTransformer(),
        chunker=CharacterChunker(
            chunk_size=300,
            chunk_overlap=50,
        ),
        embedder=OllamaEmbedder(
            model="nomic-embed-text",
        ),
    )

    vector_store = ChromaVectorStore(
        path=VECTOR_DB_DIR,
        collection_name="ragkit_examples",
    )

    indexer = DocumentIndexer(
        processor=processor,
        vector_store=vector_store,
    )

    result = indexer.index(
        source,
    )

    print()
    print("=" * 70)
    print("Index Created")
    print("=" * 70)

    print(f"Documents Indexed : {result.documents}")
    print(f"Chunks Created    : {result.chunks}")
    print(f"Embeddings Stored : {result.embeddings}")
    print()

    print(f"Vector Store      : Chroma")
    print(f"Collection        : ragkit_examples")
    print(f"Database Path     : ./vector_db")

    #
    # Verify persistence.
    #
    print()
    print("=" * 70)
    print("Verification")
    print("=" * 70)

    print(
        f"Vectors in Store  : {vector_store.count()}"
    )

    #
    # Learning recap.
    #
    print()
    print("=" * 70)
    print("What did we learn?")
    print("=" * 70)

    print("✓ Embeddings are stored in a vector database.")
    print("✓ Indexing is usually performed once.")
    print("✓ Search can be performed many times.")
    print("✓ Vector databases persist embeddings for future use.")

    #
    # Knowledge Check
    #
    print()
    print("=" * 70)
    print("Knowledge Check")
    print("=" * 70)

    print("Q1. Why don't we generate embeddings for every query?")
    print()
    print("Answer:")
    print("Generating document embeddings repeatedly is slow and")
    print("expensive. We generate them once during indexing.")
    print()

    print("Q2. What is stored inside the vector database?")
    print()
    print("Answer:")
    print("The vector database stores embeddings together with")
    print("their associated chunk metadata.")
    print()

    print("Q3. Can we search without first building an index?")
    print()
    print("Answer:")
    print("No. The vector database must contain embeddings before")
    print("similarity search can return relevant chunks.")
    print()

    print("=" * 70)
    print("Next Example")
    print("=" * 70)
    print("05_similarity_search.py")


if __name__ == "__main__":
    main()