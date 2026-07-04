"""
Purpose
-------
Represents LLM generation failures.
"""

from ragkit.exceptions.service_error import ServiceError


class LLMError(ServiceError):
    """
    Raised when text generation fails.
    """