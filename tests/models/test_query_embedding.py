from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ragkit.models.query_embedding import QueryEmbedding


def test_query_embedding_fields():
    """
    QueryEmbedding stores all constructor values.
    """

    embedding = QueryEmbedding(
        model="nomic-embed-text",
        vector=[0.1, 0.2, 0.3],
    )

    assert embedding.model == "nomic-embed-text"
    assert embedding.vector == [0.1, 0.2, 0.3]


def test_query_embedding_slots():
    """
    QueryEmbedding uses slots and therefore has no __dict__.
    """

    embedding = QueryEmbedding(
        model="test-model",
        vector=[1.0],
    )

    assert not hasattr(embedding, "__dict__")


def test_query_embedding_equality():
    """
    Two QueryEmbedding objects with identical values are equal.
    """

    left = QueryEmbedding(
        model="model",
        vector=[1.0, 2.0],
    )

    right = QueryEmbedding(
        model="model",
        vector=[1.0, 2.0],
    )

    assert left == right


def test_query_embedding_repr():
    """
    Dataclass provides a useful string representation.
    """

    embedding = QueryEmbedding(
        model="model",
        vector=[0.5],
    )

    representation = repr(embedding)

    assert "QueryEmbedding" in representation 