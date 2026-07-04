"""
Purpose
-------
Represents vector store failures.
"""

from ragkit.exceptions.ragkit_error import RagKitError


class VectorStoreError(RagKitError):
    """
    Raised when vector store operations fail.
    """