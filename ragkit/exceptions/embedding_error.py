"""
Purpose
-------
Represents embedding generation failures.
"""

from ragkit.exceptions.ragkit_error import RagKitError


class EmbeddingError(RagKitError):
    """
    Raised when embedding generation fails.
    """