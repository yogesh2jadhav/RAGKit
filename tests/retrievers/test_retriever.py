import pytest

from ragkit.retrievers.retriever import Retriever


def test_retriever_is_abstract():
    """
    Verify that Retriever cannot be instantiated
    because it is an abstract base class.
    """

    with pytest.raises(TypeError):
        Retriever()