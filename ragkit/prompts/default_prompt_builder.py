"""
Purpose
-------
Builds the default prompt for Retrieval-Augmented Generation (RAG).

Responsibilities
----------------
- Combine retrieved context into a single prompt.
- Append the user query.
- Produce a deterministic prompt for the LLM.

Does NOT
--------
- Retrieve documents.
- Generate embeddings.
- Call an LLM.
"""

from __future__ import annotations

from collections.abc import Iterable

from ragkit.models.search_result import SearchResult
from ragkit.prompts.prompt_builder import PromptBuilder

'''
=> This class implements build method of PromptBuilder interface. It just do
string join of search results
'''
class DefaultPromptBuilder(PromptBuilder):
    """
    Default implementation of PromptBuilder.
    """

    def build(
        self,
        query: str,
        search_results: Iterable[SearchResult],
    ) -> str:
        """
        Build a prompt from the supplied search results.
        """

        context = "\n\n".join(
            result.chunk.content
            for result in search_results
        )

        return f"""You are a helpful AI assistant.

Use ONLY the provided context to answer the query.

If the answer cannot be found in the context,
reply that you do not know.

Context
-------
{context}

Query
-----
{query}
"""
