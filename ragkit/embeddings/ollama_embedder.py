"""
Purpose
-------
Generate vector embeddings using a locally running Ollama embedding model.

Responsibilities
----------------
- Connect to Ollama.
- Generate embeddings for Chunk objects.
- Return Embedding domain objects.

Does NOT
--------
- Store embeddings.
- Perform similarity search.
- Generate chat responses.
"""

from __future__ import annotations
from ragkit.exceptions.embedding_error import EmbeddingError

from collections.abc import Iterable

from ollama import Client

from ragkit.embeddings.embedder import Embedder
from ragkit.models.chunk import Chunk
from ragkit.models.embedding import Embedding
from ragkit.models.query_embedding import QueryEmbedding
from ragkit.config.embedding_config import EmbeddingConfig

'''
=> This class have implemnetd Embedder interface.
   This method will connect to Ollama.
'''
class OllamaEmbedder(Embedder):
    """
    Generates embeddings using an Ollama embedding model.
    """

    '''
    => Following constructor will take model name and host url  as input and
       will connect to Model Client. Client object will be stored in _client.
    '''
    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        *,
        config: EmbeddingConfig | None = None,
    ) -> None:
        """
        Initialize the Ollama embedder.
        """

        #
        # Configuration object overrides
        # explicitly supplied values.
        #
        if config is not None:
            model = config.model

        self._model_name = model

        self._client = Client(
            host=host,
        )
    '''
    => This embed take Iterable[Chunk], reason is when we do chunk we are doing stream using yield to support that 
    we need Iterable[Chunk]. List will not work in this case.
    '''
    def embed(
        self,
        chunks: Iterable[Chunk],
    ) -> Iterable[Embedding]:
        """
        Generate embeddings for document chunks.
        """

        for chunk in chunks:
            response = self._client.embed(
                model=self._model_name,
                input=chunk.content,
            )
            if not response.embeddings:
                raise EmbeddingError("Ollama returned no embedding for document chunk.")
            yield Embedding(
                chunk_id=chunk.id,
                model=self._model_name,
                vector=response.embeddings[0],
            )

    def embed_query(
        self,
        query: str,
    ) -> QueryEmbedding:
        """
        Generate an embedding for a user query.
        """

        response = self._client.embed(
            model=self._model_name,
            input=query,
        )
        if not response.embeddings:
            raise EmbeddingError("Ollama returned no embedding for the query.")
        return QueryEmbedding(
            model=self._model_name,
            vector=response.embeddings[0],
        )