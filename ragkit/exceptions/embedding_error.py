"""
Purpose
-------
Represents embedding generation failures.
"""

from ragkit.exceptions.service_error import ServiceError


class EmbeddingError(ServiceError):
    """
    Raised when embedding generation fails.
    """