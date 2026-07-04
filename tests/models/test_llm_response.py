from ragkit.models.llm_response import LLMResponse


def test_create_llm_response():
    """
    Verify that an LLMResponse stores generated text.
    """

    response = LLMResponse(
        text="Apache Spark is a distributed computing engine.",
    )

    assert (
        response.text
        == "Apache Spark is a distributed computing engine."
    )