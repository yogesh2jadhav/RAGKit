"""
Purpose
-------
Represents vector store failures.
"""

from ragkit.exceptions.service_error import ServiceError


class VectorStoreError(ServiceError):
    """
    Raised when vector store operations fail.
    """