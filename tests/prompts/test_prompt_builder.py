import pytest

from ragkit.prompts.prompt_builder import PromptBuilder


def test_prompt_builder_is_abstract():
    """
    Verify PromptBuilder cannot be instantiated.
    """

    with pytest.raises(TypeError):
        PromptBuilder()