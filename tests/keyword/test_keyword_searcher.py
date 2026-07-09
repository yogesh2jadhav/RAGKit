import pytest

from ragkit.keyword.keyword_searcher import KeywordSearcher


def test_keyword_searcher_is_abstract():
    """
    Verify KeywordSearcher cannot be instantiated.
    """

    with pytest.raises(TypeError):
        KeywordSearcher()