import pytest

from ragkit.pipelines.pipeline import Pipeline


def test_pipeline_is_abstract():
    """
    Verify Pipeline cannot be instantiated.
    """

    with pytest.raises(TypeError):
        Pipeline()