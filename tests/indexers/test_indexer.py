import pytest

from ragkit.indexers.indexer import Indexer


def test_indexer_is_abstract():
    """
    Verify Indexer cannot be instantiated.
    """

    with pytest.raises(TypeError):
        Indexer()