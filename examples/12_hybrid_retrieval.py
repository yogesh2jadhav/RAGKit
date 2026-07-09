"""Example 12 - Hybrid Retrieval"""

from pathlib import Path
from ragkit.chunkers.character_chunker import CharacterChunker
from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.indexers.document_indexer import DocumentIndexer
from ragkit.keyword.bm25_searcher import BM25Searcher
from ragkit.processors.document_processor import DocumentProcessor
from ragkit.retrievers.hybrid_retriever import HybridRetriever
from ragkit.retrievers.similarity_retriever import SimilarityRetriever
from ragkit.sources.local_source import LocalSource
from ragkit.transformers.markdown_transformer import MarkdownTransformer
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore

EXAMPLES_DIR = Path(__file__).parent
DOCS_DIR = EXAMPLES_DIR / "docs"
DATA_DIR = EXAMPLES_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db_hybrid"
COLLECTION_NAME = "ragkit_hybrid"


def build_index(vector_store):
    processor = DocumentProcessor(
        transformer=MarkdownTransformer(),
        chunker=CharacterChunker(chunk_size=300, chunk_overlap=50),
        embedder=OllamaEmbedder(model="nomic-embed-text"),
    )
    indexer = DocumentIndexer(processor=processor, vector_store=vector_store)
    result = indexer.index(LocalSource(DOCS_DIR))
    print(result)


def print_results(title, results):
    print("=" * 70)
    print(title)
    print("=" * 70)
    for i, r in enumerate(results, 1):
        p = r.chunk.content.replace("\\n", " ")
        print(f"{i}. {r.score:.4f} {p[:100]}")


def compare(vector_store):
    embedder = OllamaEmbedder(model="nomic-embed-text")
    sim = SimilarityRetriever(embedder=embedder, vector_store=vector_store)
    kw = BM25Searcher(vector_store=vector_store)
    hybrid = HybridRetriever(retriever=sim, keyword_searcher=kw)
    while True:
        q = input("Query (back/exit/quit): ").strip()

        if q.lower() in {
            "back",
            "exit",
            "quit",
        }:
            break

        if not q:
            print()
            print("Please enter a question.")
            continue
        print_results("Similarity", sim.retrieve(query=q, top_k=3))
        print_results("BM25", list(kw.search(query=q, top_k=3)))
        print_results("Hybrid", hybrid.retrieve(query=q, top_k=3))


def statistics(vector_store):
    print(vector_store.count())


def main():
    vs = ChromaVectorStore(path=VECTOR_DB_DIR, collection_name=COLLECTION_NAME)
    while True:
        print("1.Build Index\n2.Compare Retrieval\n3.Statistics\n4.Exit")
        c = input("Select: ").strip()
        if c == "1":
            build_index(vs)
        elif c == "2":
            compare(vs)
        elif c == "3":
            statistics(vs)
        elif c == "4":
            break


if __name__ == "__main__":
    main()
