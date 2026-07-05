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

from collections.abc import Iterable

from ollama import Client

from ragkit.embeddings.embedder import Embedder
from ragkit.models.chunk import Chunk
from ragkit.models.embedding import Embedding
from ragkit.models.query_embedding import QueryEmbedding

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
    ) -> None:
        """
        Parameters
        ----------
        model:Ollama embedding model.
        host: Ollama server URL.
        """

        self._model = model
        self._client = Client(host=host)

    '''
    => This embed take Iterable[Chunk], reason is when we do chunk we are doing stream using yield to support that 
    we need Iterable[Chunk]. List will not work in this case.
    '''
    def embed(
        self,
        chunks: Iterable[Chunk],
    ) -> Iterable[Embedding]:
        """
        Notes
        -----
        Version 1 performs one request per chunk.

        Version 2 will batch multiple chunks into a
        single Ollama request.
        """

        for chunk in chunks:

            response = self._client.embed(
                model=self._model,
                input=chunk.content,
            )

            vector = response.embeddings[0]

            yield Embedding(
                chunk_id=chunk.id,
                model=self._model,
                vector=vector,
            )

    def embed_query(
        self,
        query: str,
    ) -> QueryEmbedding:
        """
        Generate an embedding for a user query.
        """

        response = self._client.embed(
            model=self._model,
            input=query,
        )

        vector = response.embeddings[0]

        return QueryEmbedding(
            model=self._model,
            vector=vector,
        )