from ragkit.models.llm_response import LLMResponse


def test_create_llm_response():
    """
    Verify that an LLMResponse stores generated text.
    """

    response = LLMResponse(
        content="Apache Spark is a distributed computing engine.",
    )

    assert (
        response.content
        == "Apache Spark is a distributed computing engine."
    )