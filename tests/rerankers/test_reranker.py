import pytest

from ragkit.rerankers.reranker import Reranker


def test_reranker_is_abstract():

    with pytest.raises(TypeError):

        Reranker()