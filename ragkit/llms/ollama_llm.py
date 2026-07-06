"""
Purpose
-------
Generates text using an Ollama Large Language Model.

Responsibilities
----------------
- Connect to an Ollama server.
- Send prompts.
- Return generated responses.

Does NOT
--------
- Retrieve documents.
- Build prompts.
- Generate embeddings.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import ollama

from ragkit.exceptions import LLMError
from ragkit.llms.llm import LLM
from ragkit.models.llm_response import LLMResponse
from ragkit.config.llm_config import LLMConfig

'''
=> OllamaLLM is class which is implement LLM interface. with generate method.
'''
class OllamaLLM(LLM):
    """
    Ollama implementation of the LLM interface.
    => Following is the constructor for this class which take model name and host url. as input and
    connect to an Ollama server.
    """
    def __init__(
        self,
        model: str = "qwen3:8b",
        host: str = "http://localhost:11434",
        *,
        config: LLMConfig | None = None,
    ) -> None:
        """
        Initialize the Ollama LLM.

        Parameters
        ----------
        model Name of the language model.
        host Ollama server URL.
        config  Optional LLM configuration.
        """

        #
        # Configuration overrides explicit parameters.
        #
        if config is not None:
            model = config.model

        self._model_name = model

        self._client = ollama.Client(  # => We are createing ollam client here.
            host=host,
        )

    def generate(
        self,
        prompt: str,
        options: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Generate a response using Ollama.
        """

        try:
            response = self._client.generate( # => where we send Prompt (all search output and question to Ollama) and get response
                model=self._model_name,
                prompt=prompt,
                options=dict(options) if options else None,
            )

            return LLMResponse(
                content=response["response"],
            )

        except Exception as ex:
            raise LLMError(
                "Failed to generate response using Ollama."
            ) from ex