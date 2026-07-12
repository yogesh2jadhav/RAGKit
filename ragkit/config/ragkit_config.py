"""
Purpose
-------
Aggregate all RAGKit configuration.

Responsibilities
----------------
- Group component configurations.
- Provide sensible defaults.

Does NOT
--------
- Create framework components.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from ragkit.config.chunker_config import ChunkerConfig
from ragkit.config.embedding_config import EmbeddingConfig
from ragkit.config.llm_config import LLMConfig
from ragkit.config.reranker_config import RerankerConfig
from ragkit.config.retrieval_config import RetrievalConfig
from ragkit.config.vector_store_config import VectorStoreConfig


@dataclass(frozen=True, slots=True)
class RagKitConfig:
    """
    Root configuration for the framework.
    """

    chunker: ChunkerConfig = field(
        default_factory=ChunkerConfig,
    )

    embedding: EmbeddingConfig = field(
        default_factory=EmbeddingConfig,
    )

    llm: LLMConfig = field(
        default_factory=LLMConfig,
    )

    retrieval: RetrievalConfig = field(
        default_factory=RetrievalConfig,
    )

    reranker: RerankerConfig = field(
        default_factory=RerankerConfig,
    )

    vector_store: VectorStoreConfig = field(
        default_factory=VectorStoreConfig,
    )