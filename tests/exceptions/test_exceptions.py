from ragkit.exceptions import (
    EmbeddingError,
    LLMError,
    RagKitError,
    VectorStoreError,
)


def test_exception_hierarchy():
    """
    Verify all framework exceptions inherit
    from RagKitError.
    """

    assert issubclass(EmbeddingError, RagKitError)
    assert issubclass(LLMError, RagKitError)
    assert issubclass(VectorStoreError, RagKitError)


def test_raise_embedding_error():
    """
    Verify EmbeddingError can be raised.
    """

    try:
        raise EmbeddingError("embedding failed")
    except RagKitError as ex:
        assert str(ex) == "embedding failed"


def test_raise_llm_error():
    """
    Verify LLMError can be raised.
    """

    try:
        raise LLMError("llm failed")
    except RagKitError as ex:
        assert str(ex) == "llm failed"


def test_raise_vector_store_error():
    """
    Verify VectorStoreError can be raised.
    """

    try:
        raise VectorStoreError("vector store failed")
    except RagKitError as ex:
        assert str(ex) == "vector store failed"