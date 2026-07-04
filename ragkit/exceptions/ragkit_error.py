"""
Purpose
-------
Defines the base exception for RAGKit.

Responsibilities
----------------
- Provide a common base class for all framework exceptions.

Does NOT
--------
- Represent provider-specific failures.
"""


class RagKitError(Exception):
    """
    Base exception for all RAGKit errors.
    """