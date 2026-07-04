"""
Purpose
-------
Represents LLM generation failures.
"""

from ragkit.exceptions.ragkit_error import RagKitError


class LLMError(RagKitError):
    """
    Raised when text generation fails.
    """