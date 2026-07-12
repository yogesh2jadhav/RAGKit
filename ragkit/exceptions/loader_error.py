"""
Purpose
-------
Exception raised when document loading fails.

Responsibilities
----------------
- Represent loader related failures.

Does NOT
--------
- Represent embedding failures.
- Represent vector store failures.
"""

from __future__ import annotations


class LoaderError(Exception):
    """
    Raised when no suitable loader exists or
    document loading cannot be completed.
    """