"""
End-to-end RAGKit demonstration.

This example shows the complete lifecycle:

1. Discover documents
2. Load documents
3. Chunk documents
4. Generate embeddings
5. Store embeddings
6. Query the vector database
7. Generate an answer using the RetrievalPipeline
"""

from pathlib import Path

from ragkit.chunkers.character_chunker import CharacterChunker
from ragkit.embeddings.ollama_embedder import OllamaEmbedder
from ragkit.llms.ollama_llm import OllamaLLM
from ragkit.loaders.loader_factory import LoaderFactory
from ragkit.pipelines.retrieval_pipeline import RetrievalPipeline
from ragkit.prompts.default_prompt_builder import DefaultPromptBuilder
from ragkit.retrievers.similarity_retriever import SimilarityRetriever
from ragkit.sources.local_source import LocalSource
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore


def build_index() -> ChromaVectorStore:
    """
    Build the vector index.
    """

    print("=" * 70)
    print("Building Vector Index")
    print("=" * 70)

    source = LocalSource("docs")

    chunker = CharacterChunker(
        chunk_size=500,
        chunk_overlap=100,
    )

    embedder = OllamaEmbedder(
        model="nomic-embed-text",
    )

    vector_store = ChromaVectorStore(
        path="./vector_db",
        collection_name="ragkit_demo",
    )

    total_documents = 0
    total_chunks = 0

    for source_document in source.discover():

        loader = LoaderFactory.get_loader(source_document)

        document = loader.load(source_document)

        total_documents += 1

        #
        # CharacterChunker returns an Iterable.
        # Materialize it because we need it twice:
        #   1. Generate embeddings.
        #   2. Store in Chroma.
        #
        chunks = list(
            chunker.chunk(document)
        )

        total_chunks += len(chunks)

        embeddings = list(
            embedder.embed(chunks)
        )

        vector_store.add(
            chunks=chunks,
            embeddings=embeddings,
        )

        print(f"Indexed: {source_document.uri}")

    print()
    print(f"Documents : {total_documents}")
    print(f"Chunks    : {total_chunks}")
    print(f"Vectors   : {vector_store.count()}")
    print()

    return vector_store


def build_pipeline(
    vector_store: ChromaVectorStore,
) -> RetrievalPipeline:
    """
    Build the RetrievalPipeline.
    """

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

    return RetrievalPipeline(
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
    )


def main() -> None:

    if not Path("docs").exists():
        raise FileNotFoundError(
            "examples/data directory not found."
        )

    vector_store = build_index()

    pipeline = build_pipeline(
        vector_store,
    )

    print("=" * 70)
    print("RAGKit Demo")
    print("=" * 70)

    while True:

        query = input("\nQuestion (or 'exit'): ").strip()

        if query.lower() in {
            "exit",
            "quit",
        }:
            break

        response = pipeline.invoke(
            query=query,
        )

        print()
        print("-" * 70)
        print("Answer")
        print("-" * 70)
        print(response.content)
        print("-" * 70)


if __name__ == "__main__":
    main()