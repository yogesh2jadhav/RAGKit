"""
Example 13 - Metadata Filtering
Version : v01

This version builds the index and provides
helper methods. Interactive retrieval will
be added in v02.
"""

from pathlib import Path

from ragkit.chunkers.character_chunker import CharacterChunker
from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.indexers.document_indexer import DocumentIndexer
from ragkit.processors.document_processor import DocumentProcessor
from ragkit.retrievers.similarity_retriever import SimilarityRetriever
from ragkit.sources.metadata_local_source import MetadataLocalSource
from ragkit.transformers.markdown_transformer import MarkdownTransformer
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore

EXAMPLES_DIR = Path(__file__).parent

DATA_DIR = EXAMPLES_DIR / "data"

DOCS_DIR = DATA_DIR / "metadata"

VECTOR_DB_DIR = DATA_DIR / "vector_db_metadata"

COLLECTION_NAME = "ragkit_metadata"


FILE_METADATA = {
    "spark_beginner.txt": {
        "category": "spark",
        "level": "beginner",
    },
    "spark_advanced.txt": {
        "category": "spark",
        "level": "advanced",
    },
    "python_beginner.txt": {
        "category": "python",
        "level": "beginner",
    },
    "docker_intermediate.txt": {
        "category": "docker",
        "level": "intermediate",
    },
    "kubernetes_advanced.txt": {
        "category": "kubernetes",
        "level": "advanced",
    },
}


def build_index(
    vector_store: ChromaVectorStore,
) -> None:
    """
    Index every document.
    """

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

    source = MetadataLocalSource(
        directory=DOCS_DIR,
        metadata=FILE_METADATA,
    )

    result = indexer.index(
        source,
    )

    print()
    print("=" * 70)
    print("Index Build Completed")
    print("=" * 70)
    print(result)
    print()


def print_results(
    title,
    results,
):


    """
    Print search results.
    """

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    results = list(results)

    if not results:
        print("No results found.")
        print()
        return

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = result.chunk.metadata

        print(f"{index}. Score : {result.score:.4f}")

        print(
            "   Category :",
            metadata.get(
                "category",
                "-",
            ),
        )

        print(
            "   Level    :",
            metadata.get(
                "level",
                "-",
            ),
        )

        print(
            "   File     :",
            metadata.get(
                "filename",
                "-",
            ),
        )

        print(
            "   Content  :",
            result.chunk.content[:100].replace(
                "\n",
                " ",
            ),
        )

        print()


def statistics(
    vector_store: ChromaVectorStore,
) -> None:
    """
    Display collection statistics.
    """

    print()
    print("=" * 70)
    print("Statistics")
    print("=" * 70)

    print(
        "Vectors :",
        vector_store.count(),
    )

    print()


def main():

    print("=" * 70)
    print("Example 13 - Metadata Filtering")
    print("Version 02")
    print("=" * 70)

    vector_store = ChromaVectorStore(
        path=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME,
    )

    build_index(
        vector_store,
    )

    while True:

        print()

        print("1.Compare Retrieval")
        print("2.Statistics")
        print("3.Exit")

        choice = input(
            "Select: ",
        ).strip()

        if choice == "1":

            compare(
                vector_store,
            )

        elif choice == "2":

            statistics(
                vector_store,
            )

        elif choice == "3":

            break

        else:

            print("Invalid selection.")

def compare(
    vector_store: ChromaVectorStore,
) -> None:
    """
    Compare retrieval with and without metadata filters.
    """

    retriever = SimilarityRetriever(
        embedder=OllamaEmbedder(
            model="nomic-embed-text",
        ),
        vector_store=vector_store,
    )

    while True:

        print()

        query = input(
            "Query (back/exit/quit): ",
        ).strip()

        if query.lower() in {
            "back",
            "exit",
            "quit",
        }:
            break

        if not query:
            print("Please enter a query.")
            continue

        print_results(
            "Without Metadata Filter",
            retriever.retrieve(
                query=query,
                top_k=3,
            ),
        )

        print()

        category = input(
            "Category (blank = any): ",
        ).strip()

        level = input(
            "Level (blank = any): ",
        ).strip()

        filters = {}

        if category:
            filters["category"] = category

        if level:
            filters["level"] = level

        print_results(
            "With Metadata Filter",
            retriever.retrieve(
                query=query,
                top_k=3,
                filters=filters or None,
            ),
        )
        
if __name__ == "__main__":
    main()