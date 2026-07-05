"""
Purpose
-------
Executes a Retrieval-Augmented Generation (RAG) pipeline.

Responsibilities
----------------
- Retrieve relevant search results.
- Build a prompt.
- Generate an LLM response.

Does NOT
--------
- Perform retrieval logic.
- Build prompts.
- Generate embeddings.
"""

from __future__ import annotations

from ragkit.llms.llm import LLM
from ragkit.models.llm_response import LLMResponse
from ragkit.pipelines.pipeline import Pipeline
from ragkit.prompts.prompt_builder import PromptBuilder
from ragkit.retrievers.retriever import Retriever


'''
=> In this end to end execution happen
    query -> Retriever.retrieve() -> Embedder.embed_query() -> QueryEmbedding -> 
    VectorStore.search() -> SearchResult -> PromptBuilder.build() -> Prompt -> LLM.generate() -> LLMResponse
'''
class RetrievalPipeline(Pipeline):
    """
    Default Retrieval-Augmented Generation pipeline.
    """

    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm: LLM,
    ) -> None:
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm = llm

    def invoke(
        self,
        query: str,
    ) -> LLMResponse:
        """
        Execute the retrieval pipeline.
        """

        search_results = self._retriever.retrieve(
            query=query,
        )

        prompt = self._prompt_builder.build(
            query=query,
            search_results=search_results,
        )

        return self._llm.generate(
            prompt=prompt,
        )