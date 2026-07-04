from __future__ import annotations

import pytest

from ragkit.vectorstores.vector_store import VectorStore


def test_vector_store_is_abstract():
    """
    The abstract VectorStore cannot be instantiated.
    """

    with pytest.raises(TypeError):
        VectorStore()