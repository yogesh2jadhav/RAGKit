"""
Purpose
-------
Root configuration for RAGKit.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragkit.config.embedding_config import EmbeddingConfig
from ragkit.config.llm_config import LLMConfig
from ragkit.config.retrieval_config import RetrievalConfig
from ragkit.config.vector_store_config import VectorStoreConfig

# => this is main config class which hold or collect all config into one
@dataclass(frozen=True, slots=True)
class RagKitConfig:
    """
    Root configuration object.
    """

    embedding: EmbeddingConfig
    llm: LLMConfig
    retrieval: RetrievalConfig = RetrievalConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()