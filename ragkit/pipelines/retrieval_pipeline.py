
"""
Purpose
-------
End-to-end Retrieval-Augmented Generation (RAG) pipeline.

Responsibilities
----------------
- Retrieve relevant search results.
- Rerank retrieved results.
- Build an LLM prompt.
- Generate the final response.

Does NOT
--------
- Load documents.
- Generate embeddings.
- Store vectors.
"""

from __future__ import annotations

from ragkit.llms.llm import LLM
from ragkit.models.llm_response import LLMResponse
from ragkit.pipelines.pipeline import Pipeline
from ragkit.prompts.prompt_builder import PromptBuilder
from ragkit.rerankers.identity_reranker import IdentityReranker
from ragkit.rerankers.reranker import Reranker
from ragkit.retrievers.retriever import Retriever


'''
=> In this end to end execution happen
    query -> Retriever.retrieve() -> _reranker.rerank() -> Prompt -> LLM.generate() -> LLMResponse
'''
class RetrievalPipeline(Pipeline):
    """
    Default Retrieval-Augmented Generation pipeline.
    """

    def __init__(
        self,
        *,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm: LLM,
        reranker: Reranker | None = None,
    ) -> None:
        """
        Initialize the pipeline.
        """

        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm = llm

        if reranker is None:
            reranker = IdentityReranker()

        self._reranker = reranker

    def invoke(
        self,
        query: str,
    ) -> LLMResponse:
        """
        Execute the RAG pipeline.
        """

        #
        # Step 1
        # Retrieve relevant chunks.
        #
        search_results = list(
            self._retriever.retrieve(
                query=query,
            )
        )

        #
        # Step 2
        # Improve ordering.
        #
        search_results = self._reranker.rerank(
            query=query,
            results=search_results,
        )

        #
        # Step 3
        # Build prompt.
        #
        prompt = self._prompt_builder.build(
            query=query,
            search_results=search_results,
        )

        #
        # Step 4
        # Generate answer.
        #
        return self._llm.generate(
            prompt=prompt,
        )