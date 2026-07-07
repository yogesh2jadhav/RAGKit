"""
Example 09
----------

Improve retrieval quality using an LLM reranker.

Concepts
--------
- Retriever
- Search Results
- LLMReranker

Run
---
python examples/09_reranker.py

Previous
--------
08_transformers.py

Next
----
10_chat.py
"""

from pathlib import Path

from ragkit.config.reranker_config import RerankerConfig
from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.llms.ollama_llm import OllamaLLM
from ragkit.rerankers.identity_reranker import IdentityReranker
from ragkit.rerankers.llm_reranker import LLMReranker
from ragkit.retrievers.similarity_retriever import SimilarityRetriever
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore


EXAMPLES_DIR = Path(__file__).parent
DATA_DIR = EXAMPLES_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"


def print_results(
    title: str,
    results,
) -> None:

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print()

        print(f"Rank  : {rank}")
        print(f"Score : {result.score:.4f}")

        preview = result.chunk.content

        if len(preview) > 200:
            preview = preview[:200] + "..."

        print(preview)

        print("-" * 70)


def main() -> None:

    print("=" * 70)
    print("Example 09 - LLM Reranker")
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

    llm = OllamaLLM(
        model="qwen3:8b",
    )

    reranker = LLMReranker(
        llm=llm,
        config=RerankerConfig(),
    )

    identity = IdentityReranker()

    query = input(
        "\nQuestion: "
    ).strip()

    #
    # Retrieve results.
    #
    retrieved = list(
        retriever.retrieve(
            query=query,
            top_k=5,
        )
    )

    print_results(
        "Original Search Results",
        retrieved,
    )

    #
    # Identity reranker.
    #
    identity_results = identity.rerank(
        query=query,
        results=retrieved,
    )

    #
    # LLM reranker.
    #
    reranked = reranker.rerank(
        query=query,
        results=retrieved,
    )

    print_results(
        "LLM Reranked Results",
        reranked,
    )

    print()

    print("=" * 70)
    print("What changed?")
    print("=" * 70)

    print("The vector database returned results")
    print("ordered by vector similarity.")

    print()

    print("The LLM reviewed those results and")
    print("reordered them according to")
    print("their semantic relevance to")
    print("the user's question.")

    print()

    print("=" * 70)
    print("What did we learn?")
    print("=" * 70)

    print("✓ Retrieval and reranking are different steps.")
    print("✓ Vector search is fast.")
    print("✓ LLM reranking is slower but often more accurate.")
    print("✓ Reranking changes the order of results.")
    print("✓ The content of the chunks does not change.")

    print()

    print("=" * 70)
    print("Knowledge Check")
    print("=" * 70)

    print("Q1. Does reranking create new chunks?")
    print()
    print("Answer:")
    print("No. It only changes their order.")
    print()

    print("Q2. Why not always use the LLM?")
    print()
    print("Answer:")
    print("Searching every document with an LLM")
    print("would be much slower and more expensive.")
    print()

    print("Q3. Why perform vector search first?")
    print()
    print("Answer:")
    print("Vector search quickly reduces thousands")
    print("of chunks to a small candidate set.")
    print()

    print("=" * 70)
    print("Next Example")
    print("=" * 70)

    print("10_chat.py")


if __name__ == "__main__":
    main()