"""
Purpose
-------
Default transformer that leaves a document unchanged.

Responsibilities
----------------
- Return the original document.

Does NOT
--------
- Modify document content.
"""

from __future__ import annotations

from ragkit.models.document import Document
from ragkit.transformers.transformer import Transformer


class IdentityTransformer(Transformer):
    """
    Returns the supplied document unchanged.
    """

    def transform(
        self,
        document: Document,
    ) -> Document:

        return document