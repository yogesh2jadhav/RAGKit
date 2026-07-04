from __future__ import annotations

import pytest

from ragkit.vectorstores.vector_store import VectorStore


def test_vector_store_is_abstract():
    """
    The abstract VectorStore cannot be instantiated.
    Verify that VectorStore cannot be instantiated
    because it is an abstract base class.
    """

    with pytest.raises(TypeError):
        VectorStore()