from ragkit.models.indexing_result import IndexingResult


def test_create_indexing_result():
    """
    Verify an IndexingResult stores indexing statistics.
    """

    result = IndexingResult(
        documents=5,
        chunks=37,
        embeddings=37,
    )

    assert result.documents == 5
    assert result.chunks == 37
    assert result.embeddings == 37