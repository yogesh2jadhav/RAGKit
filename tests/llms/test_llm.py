import pytest

from ragkit.llms.llm import LLM


def test_llm_is_abstract():
    """
    Verify that LLM cannot be instantiated.
    """

    with pytest.raises(TypeError):
        LLM()