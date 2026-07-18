from __future__ import annotations

import pytest

from ragkit.config.vector_store_config import VectorStoreConfig
from ragkit.vectorstores.chroma_vector_store import ChromaVectorStore
from ragkit.vectorstores.vector_store import VectorStore


def test_vector_store_is_abstract():
    """
    The abstract VectorStore cannot be instantiated.
    Verify that VectorStore cannot be instantiated
    because it is an abstract base class.
    """

    with pytest.raises(TypeError):
        VectorStore()

def test_create_vector_store_with_defaults(tmp_path):
    """
    Verify the default constructor works.
    """

    store = ChromaVectorStore(
        path=str(tmp_path),
    )

    assert store.count() == 0

def test_create_vector_store_with_config(tmp_path):
    """
    Verify VectorStoreConfig is supported.
    """

    config = VectorStoreConfig(
        path=str(tmp_path),
        collection_name="unit_test",
    )

    store = ChromaVectorStore(
        config=config,
    )

    assert store.count() == 0