"""
Example 06
----------

Execute a complete Retrieval-Augmented Generation (RAG) pipeline.

Concepts
--------
- RetrievalPipeline
- PromptBuilder
- Retriever
- LLM

Run
---
python examples/06_retrieval_pipeline.py

Previous
--------
05_similarity_search.py

Next
----
07_document_indexer.py
"""

from pathlib import Path

from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.llms.ollama_llm import OllamaLLM
from ragkit.pipelines.retrieval_pipeline import RetrievalPipeline
from ragkit.prompts.default_prompt_builder import DefaultPromptBuilder
from ragkit.retrievers.similarity_retriever import SimilarityRetriever
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore

EXAMPLES_DIR = Path(__file__).parent
DATA_DIR = EXAMPLES_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"


def main() -> None:
    """
    Execute a Retrieval-Augmented Generation pipeline.
    """

    print("=" * 70)
    print("Example 06 - Retrieval Pipeline")
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

    prompt_builder = DefaultPromptBuilder()

    llm = OllamaLLM(
        model="qwen3:8b",
    )

    pipeline = RetrievalPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
    )

    query = input(
        "\nQuestion: "
    ).strip()

    #
    # Show retrieval results first.
    #
    results = list(
        retriever.retrieve(
            query=query,
            top_k=3,
        )
    )

    print()

    print("=" * 70)
    print("Retrieved Context")
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

        print("-" * 70)

    #
    # Build prompt so the learner can see it.
    #
    prompt = prompt_builder.build(
        query=query,
        search_results=results,
    )

    print()

    print("=" * 70)
    print("Prompt Sent To LLM")
    print("=" * 70)

    print(prompt)

    #
    # Execute the pipeline.
    #
    response = pipeline.invoke(
        query=query,
    )

    print()

    print("=" * 70)
    print("LLM Response")
    print("=" * 70)

    print(response.content)

    #
    # Summary
    #
    print()

    print("=" * 70)
    print("Summary")
    print("=" * 70)

    print("Question :", query)

    print()

    print("=" * 70)
    print("What did we learn?")
    print("=" * 70)

    print("✓ RetrievalPipeline orchestrates the entire RAG workflow.")
    print("✓ The retriever finds relevant chunks.")
    print("✓ PromptBuilder combines the retrieved context with the query.")
    print("✓ The LLM generates an answer using the supplied context.")
    print("✓ The LLM never searches the vector database directly.")

    print()

    print("=" * 70)
    print("Knowledge Check")
    print("=" * 70)

    print("Q1. What is the responsibility of RetrievalPipeline?")
    print()
    print("Answer:")
    print("It coordinates the retriever, prompt builder and LLM.")
    print()

    print("Q2. Does the LLM retrieve documents?")
    print()
    print("Answer:")
    print("No. Retrieval is performed by the Retriever.")
    print()

    print("Q3. Why do we build a prompt?")
    print()
    print("Answer:")
    print("The prompt provides the retrieved context together")
    print("with the user's question so the LLM can answer accurately.")
    print()

    print("=" * 70)
    print("Next Example")
    print("=" * 70)
    print("07_document_indexer.py")


if __name__ == "__main__":
    main()