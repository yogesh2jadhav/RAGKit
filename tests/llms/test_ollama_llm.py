from unittest.mock import MagicMock
from unittest.mock import patch

from ragkit.exceptions import LLMError
from ragkit.llms.ollama_llm import OllamaLLM


@patch("ragkit.llms.ollama_llm.ollama.Client")
def test_generate(mock_client):
    """
    Verify OllamaLLM generates a response.
    """

    client = MagicMock()

    client.generate.return_value = {
        "response": "Hello World",
    }

    mock_client.return_value = client

    llm = OllamaLLM(
        model="llama3.2",
    )

    response = llm.generate(
        prompt="Say hello",
    )

    assert response.content == "Hello World"

    client.generate.assert_called_once_with(
        model="llama3.2",
        prompt="Say hello",
        options=None,
    )


@patch("ragkit.llms.ollama_llm.ollama.Client")
def test_generate_with_options(mock_client):
    """
    Verify generation options are forwarded.
    """

    client = MagicMock()

    client.generate.return_value = {
        "response": "Hello",
    }

    mock_client.return_value = client

    llm = OllamaLLM(
        model="llama3.2",
    )

    llm.generate(
        prompt="Hello",
        options={
            "temperature": 0,
        },
    )

    client.generate.assert_called_once_with(
        model="llama3.2",
        prompt="Hello",
        options={
            "temperature": 0,
        },
    )


@patch("ragkit.llms.ollama_llm.ollama.Client")
def test_generate_failure(mock_client):
    """
    Verify provider exceptions are wrapped.
    """

    client = MagicMock()

    client.generate.side_effect = RuntimeError(
        "Connection failed",
    )

    mock_client.return_value = client

    llm = OllamaLLM(
        model="llama3.2",
    )

    try:
        llm.generate(
            prompt="Hello",
        )
        assert False

    except LLMError as ex:
        assert str(ex) == (
            "Failed to generate response using Ollama."
        )