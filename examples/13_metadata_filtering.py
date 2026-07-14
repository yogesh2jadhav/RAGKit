"""
Example 13
----------

Metadata Filtering

Demonstrates how metadata filtering limits
retrieval to matching documents.
"""

from __future__ import annotations

from pathlib import Path

from ragkit.chunkers.character_chunker import CharacterChunker
from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.indexers.document_indexer import DocumentIndexer
from ragkit.loaders.loader_factory import LoaderFactory
from ragkit.models.search_result import SearchResult
from ragkit.models.source_document import SourceDocument
from ragkit.processors.document_processor import DocumentProcessor
from ragkit.retrievers.similarity_retriever import SimilarityRetriever
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore


def create_sources() -> list[SourceDocument]:
    base = Path("examples/data/metadata")
    return [
        SourceDocument(uri=str(base / "spark_beginner.txt"), mime_type="text/plain", metadata={"category": "spark", "level": "beginner"}),
        SourceDocument(uri=str(base / "spark_advanced.txt"), mime_type="text/plain", metadata={"category": "spark", "level": "advanced"}),
        SourceDocument(uri=str(base / "python_beginner.txt"), mime_type="text/plain", metadata={"category": "python", "level": "beginner"}),
        SourceDocument(uri=str(base / "docker_intermediate.txt"), mime_type="text/plain", metadata={"category": "docker", "level": "intermediate"}),
        SourceDocument(uri=str(base / "kubernetes_advanced.txt"), mime_type="text/plain", metadata={"category": "kubernetes", "level": "advanced"}),
    ]


def print_header() -> None:
    print("=" * 70)
    print("Example 13 - Metadata Filtering")
    print("=" * 70)


def print_results(results: list[SearchResult]) -> None:
    if not results:
        print("\\nNo matching documents.\\n")
        return
    for index, result in enumerate(results, start=1):
        print("-" * 70)
        print(f"Result   : {index}")
        print(f"Score    : {result.score:.4f}")
        print(f"Category : {result.chunk.metadata.get('category')}")
        print(f"Level    : {result.chunk.metadata.get('level')}")
        print()
        print(result.chunk.content[:300])
        print()


def main() -> None:
    print_header()
    print("\\nSkeleton created successfully.")
    print("Next step: build the index and add the interactive menu.")


if __name__ == "__main__":
    main()
