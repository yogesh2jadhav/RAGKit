"""
Purpose
-------
Normalizes whitespace inside documents.

Responsibilities
----------------
- Collapse repeated spaces.
- Collapse repeated blank lines.
- Trim leading/trailing whitespace.

Does NOT
--------
- Remove Markdown.
- Remove HTML.
"""

from __future__ import annotations

import re

from ragkit.models.document import Document
from ragkit.transformers.transformer import Transformer


class WhitespaceTransformer(Transformer):
    """
    Normalizes whitespace.
    """

    _MULTIPLE_SPACES = re.compile(
        r"[ \t]+",
    )

    _MULTIPLE_BLANK_LINES = re.compile(
        r"\n{3,}",
    )

    def transform(
        self,
        document: Document,
    ) -> Document:

        content = document.content

        #
        # Collapse spaces and tabs.
        #
        content = self._MULTIPLE_SPACES.sub(
            " ",
            content,
        )

        #
        # Collapse excessive blank lines.
        #
        content = self._MULTIPLE_BLANK_LINES.sub(
            "\n\n",
            content,
        )

        content = content.strip()

        return Document(
            id=document.id,
            content=content,
            metadata=document.metadata,
        )