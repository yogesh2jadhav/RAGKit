"""
Example 05
----------

Perform similarity search using a vector database.

Concepts
--------
- Query Embedding
- Similarity Search
- SearchResult
- Similarity Score

Run
---
python examples/05_similarity_search.py

Previous
--------
04_build_vector_index.py

Next
----
06_retrieval_pipeline.py
"""

from pathlib import Path

from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.retrievers.similarity_retriever import SimilarityRetriever
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore

EXAMPLES_DIR = Path(__file__).parent
DATA_DIR = EXAMPLES_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"


def main() -> None:

    print("=" * 70)
    print("Example 05 - Similarity Search")
    print("=" * 70)

    vector_store = ChromaVectorStore(
        path=VECTOR_DB_DIR,
        collection_name="ragkit_examples",
    )

    embedder = OllamaEmbedder(
        model="nomic-embed-text",
    )

    retriever = SimilarityRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    query = input(
        "\nQuestion: "
    ).strip()

    print()

    print("Generating query embedding...")

    print("Searching vector database...")

    results = list(
        retriever.retrieve(
            query=query,
            top_k=3,
        )
    )

    print()

    print("=" * 70)
    print("Retrieved Results")
    print("=" * 70)

    if not results:

        print("No matching chunks found.")

        return

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print()

        print(f"Rank  : {rank}")

        print(f"Score : {result.score:.4f}")

        print()

        preview = result.chunk.content

        if len(preview) > 300:

            preview = preview[:300] + "..."

        print(preview)

        print()

        print("-" * 70)

    print()

    print("=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"Query            : {query}")

    print(f"Results Returned : {len(results)}")

    print()

    print("=" * 70)
    print("What did we learn?")
    print("=" * 70)

    print("✓ A user query is converted into an embedding.")
    print("✓ The vector database compares embeddings, not raw text.")
    print("✓ Similarity search returns relevant chunks.")
    print("✓ The vector database does not generate answers.")

    print()

    print("=" * 70)
    print("Knowledge Check")
    print("=" * 70)

    print("Q1. Did the vector database answer the question?")

    print()

    print("Answer:")

    print("No. It only retrieved the most relevant chunks.")

    print()

    print("Q2. Why are similarity scores important?")

    print()

    print("Answer:")

    print("They indicate how closely a chunk matches the query.")

    print()

    print("Q3. Why retrieve multiple chunks instead of one?")

    print()

    print("Answer:")

    print("Relevant information is often spread across several chunks.")

    print()

    print("=" * 70)
    print("Next Example")
    print("=" * 70)

    print("06_retrieval_pipeline.py")


if __name__ == "__main__":
    main()