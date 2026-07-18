"""
Example 11
----------

Complete Retrieval-Augmented Generation application.

Concepts
--------
- Document Indexing
- Similarity Search
- Retrieval Pipeline
- Interactive Chat

Run
---
python examples/11_complete_rag.py

Previous
--------
10_chat.py
"""

from pathlib import Path

from ragkit.chunkers.character_chunker import CharacterChunker
from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.indexers.document_indexer import DocumentIndexer
from ragkit.llms.ollama_llm import OllamaLLM
from ragkit.pipelines.retrieval_pipeline import RetrievalPipeline
from ragkit.processors.document_processor import DocumentProcessor
from ragkit.prompts.default_prompt_builder import DefaultPromptBuilder
from ragkit.retrievers.similarity_retriever import SimilarityRetriever
from ragkit.sources.local_source import LocalSource
from ragkit.transformers.markdown_transformer import MarkdownTransformer
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore

EXAMPLES_DIR = Path(__file__).parent
DOCS_DIR = EXAMPLES_DIR / "docs"
DATA_DIR = EXAMPLES_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

COLLECTION_NAME = "ragkit_examples"


def build_index(vector_store: ChromaVectorStore) -> None:
    """
    Build the vector index.
    """

    print()
    print("Building vector index...")
    print()

    processor = DocumentProcessor(
        transformer=MarkdownTransformer(),
        chunker=CharacterChunker(
            chunk_size=300,
            chunk_overlap=50,
        ),
        embedder=OllamaEmbedder(
            model="nomic-embed-text",
        ),
    )

    indexer = DocumentIndexer(
        processor=processor,
        vector_store=vector_store,
    )

    result = indexer.index(
        LocalSource(DOCS_DIR),
    )

    print()

    print("Index completed.")

    print()

    print(f"Documents : {result.documents}")
    print(f"Chunks    : {result.chunks}")
    print(f"Vectors   : {result.embeddings}")


def chat(vector_store: ChromaVectorStore) -> None:
    """
    Interactive chat.
    """

    retriever = SimilarityRetriever(
        embedder=OllamaEmbedder(
            model="nomic-embed-text",
        ),
        vector_store=vector_store,
    )

    pipeline = RetrievalPipeline(
        retriever=retriever,
        prompt_builder=DefaultPromptBuilder(),
        llm=OllamaLLM(
            model="qwen3:8b",
        ),
    )

    print()

    print("Enter 'back' to return to the menu.")

    while True:

        print()

        query = input("You : ").strip()

        if query.lower() == "back":
            break

        if not query:
            continue

        print()

        response = pipeline.invoke(
            query=query,
        )

        print("Assistant")
        print("-" * 70)
        print(response.content)


def statistics(vector_store: ChromaVectorStore) -> None:
    """
    Display vector store statistics.
    """

    print()

    print("=" * 70)
    print("Knowledge Base Statistics")
    print("=" * 70)

    print(f"Stored Vectors : {vector_store.count()}")
    print(f"Collection     : {COLLECTION_NAME}")
    print(f"Database Path  : {VECTOR_DB_DIR}")


def main() -> None:

    vector_store = ChromaVectorStore(
        path=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME,
    )

    while True:

        print()
        print("=" * 70)
        print("RAGKit Complete Demo")
        print("=" * 70)

        print("1. Build Index")
        print("2. Chat")
        print("3. Statistics")
        print("4. Exit")

        print()

        choice = input("Select option: ").strip()

        if choice == "1":

            build_index(
                vector_store,
            )

        elif choice == "2":

            chat(
                vector_store,
            )

        elif choice == "3":

            statistics(
                vector_store,
            )

        elif choice == "4":

            print()

            print("Thank you for learning RAGKit.")

            break

        else:

            print()

            print("Invalid option.")


if __name__ == "__main__":
    main()