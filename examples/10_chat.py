"""
Example 10
----------

Interactive RAG Chat.

Concepts
--------
- RetrievalPipeline
- Interactive Chat
- Continuous Question Answering

Run
---
python examples/10_chat.py

Previous
--------
09_reranker.py

Next
----
11_complete_rag.py
"""

from pathlib import Path

from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.llms.ollama_llm import OllamaLLM
from ragkit.pipelines.retrieval_pipeline import RetrievalPipeline
from ragkit.prompts.default_prompt_builder import DefaultPromptBuilder
from ragkit.rerankers.identity_reranker import IdentityReranker
from ragkit.retrievers.similarity_retriever import SimilarityRetriever
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore

EXAMPLES_DIR = Path(__file__).parent
DATA_DIR = EXAMPLES_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"


def main() -> None:
    """
    Interactive RAG chat.
    """

    print("=" * 70)
    print("Example 10 - Interactive RAG Chat")
    print("=" * 70)

    print()
    print("Type your questions.")
    print("Type 'exit' to quit.")
    print()

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
        reranker=IdentityReranker(),
        prompt_builder=prompt_builder,
        llm=llm,
    )

    while True:

        print()
        query = input("You : ").strip()

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            print()
            print("Goodbye!")
            break

        print()
        print("Searching knowledge base...")
        print()

        try:

            response = pipeline.invoke(
                query=query,
            )

            print("Assistant")
            print("-" * 70)
            print(response.content)

        except Exception as ex:

            print()
            print("An error occurred.")
            print(ex)


if __name__ == "__main__":
    main()