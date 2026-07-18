"""
Example 07
----------

Index documents using DocumentIndexer.

Concepts
--------
- DocumentProcessor
- DocumentIndexer

Run
---
python examples/07_document_indexer.py

Previous
--------
06_retrieval_pipeline.py

Next
----
08_transformers.py
"""

from pathlib import Path

from ragkit.chunkers.character_chunker import CharacterChunker
from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.indexers.document_indexer import DocumentIndexer
from ragkit.processors.document_processor import DocumentProcessor
from ragkit.sources.local_source import LocalSource
from ragkit.transformers.identity_transformer import IdentityTransformer
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore

EXAMPLES_DIR = Path(__file__).parent
DOCS_DIR = EXAMPLES_DIR / "docs"
DATA_DIR = EXAMPLES_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"


def main() -> None:
    """
    Demonstrate DocumentIndexer.
    """

    print("=" * 70)
    print("Example 07 - Document Indexer")
    print("=" * 70)

    print()
    print("Creating document source...")

    source = LocalSource(
        DOCS_DIR,
    )

    print("Creating document processor...")

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

    print("Opening vector store...")

    vector_store = ChromaVectorStore(
        path=VECTOR_DB_DIR,
        collection_name="ragkit_examples",
    )

    print("Creating document indexer...")

    indexer = DocumentIndexer(
        processor=processor,
        vector_store=vector_store,
    )

    print()
    print("Indexing documents...")
    print()

    result = indexer.index(
        source,
    )

    print("=" * 70)
    print("Indexing Completed")
    print("=" * 70)

    print(f"Documents Indexed : {result.documents}")
    print(f"Chunks Created    : {result.chunks}")
    print(f"Embeddings Stored : {result.embeddings}")
    print(f"Vectors In Store  : {vector_store.count()}")

    print()

    print("=" * 70)
    print("What happened internally?")
    print("=" * 70)

    print("✓ Source discovered documents.")
    print("✓ Appropriate loader selected for each document.")
    print("✓ Documents transformed.")
    print("✓ Documents split into chunks.")
    print("✓ Embeddings generated.")
    print("✓ Embeddings stored in ChromaDB.")

    print()

    print("=" * 70)
    print("Why use DocumentIndexer?")
    print("=" * 70)

    print("Without DocumentIndexer you would manually:")
    print()

    print("  • Discover documents")
    print("  • Select a loader")
    print("  • Load documents")
    print("  • Transform documents")
    print("  • Chunk documents")
    print("  • Generate embeddings")
    print("  • Store embeddings")

    print()

    print("DocumentIndexer performs all of these steps")
    print("using a single method call:")

    print()

    print("    indexer.index(source)")

    print()

    print("=" * 70)
    print("Knowledge Check")
    print("=" * 70)

    print("Q1. Does DocumentIndexer generate embeddings?")
    print()
    print("Answer:")
    print("Yes. It delegates that work to DocumentProcessor.")
    print()

    print("Q2. Does DocumentIndexer perform vector search?")
    print()
    print("Answer:")
    print("No. It only builds the vector index.")
    print()

    print("Q3. What is the benefit of DocumentIndexer?")
    print()
    print("Answer:")
    print("It orchestrates the complete indexing workflow")
    print("using a single API call.")

    print()

    print("=" * 70)
    print("Next Example")
    print("=" * 70)

    print("08_transformers.py")


if __name__ == "__main__":
    main()