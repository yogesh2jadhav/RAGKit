import pytest

from ragkit.transformers.transformer import Transformer


def test_transformer_is_abstract():

    with pytest.raises(TypeError):

        Transformer()