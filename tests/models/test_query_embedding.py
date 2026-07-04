from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ragkit.models.query_embedding import QueryEmbedding


def test_query_embedding_fields():
    """
    QueryEmbedding stores all constructor values.
    """

    embedding = QueryEmbedding(
        content="What is Python?",
        model="nomic-embed-text",
        vector=[0.1, 0.2, 0.3],
    )

    assert embedding.content == "What is Python?"
    assert embedding.model == "nomic-embed-text"
    assert embedding.vector == [0.1, 0.2, 0.3]


def test_query_embedding_is_frozen():
    """
    QueryEmbedding is immutable.
    """

    embedding = QueryEmbedding(
        content="Question",
        model="test-model",
        vector=[1.0],
    )

    with pytest.raises(FrozenInstanceError):
        embedding.content = "Modified"


def test_query_embedding_slots():
    """
    QueryEmbedding uses slots and therefore has no __dict__.
    """

    embedding = QueryEmbedding(
        content="Question",
        model="test-model",
        vector=[1.0],
    )

    assert not hasattr(embedding, "__dict__")


def test_query_embedding_equality():
    """
    Two QueryEmbedding objects with identical values are equal.
    """

    left = QueryEmbedding(
        content="Question",
        model="model",
        vector=[1.0, 2.0],
    )

    right = QueryEmbedding(
        content="Question",
        model="model",
        vector=[1.0, 2.0],
    )

    assert left == right


def test_query_embedding_repr():
    """
    Dataclass provides a useful string representation.
    """

    embedding = QueryEmbedding(
        content="Hello",
        model="model",
        vector=[0.5],
    )

    representation = repr(embedding)

    assert "QueryEmbedding" in representation
    assert "Hello" in representation