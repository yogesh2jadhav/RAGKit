"""
Purpose
-------
Transforms Markdown documents into plain text.

Responsibilities
----------------
- Remove Markdown syntax.
- Preserve readable text.

Does NOT
--------
- Collapse whitespace.
- Chunk documents.
- Generate embeddings.
"""

from __future__ import annotations

import re

from ragkit.models.document import Document
from ragkit.transformers.transformer import Transformer


class MarkdownTransformer(Transformer):
    """
    Removes common Markdown syntax while preserving
    readable content.
    """

    #
    # Precompiled regular expressions.
    #
    _CODE_BLOCK = re.compile(
        r"```.*?```",
        flags=re.DOTALL,
    )

    _INLINE_CODE = re.compile(
        r"`([^`]*)`",
    )

    _IMAGE = re.compile(
        r"!\[[^\]]*]\([^)]+\)",
    )

    _LINK = re.compile(
        r"\[([^\]]+)]\([^)]+\)",
    )

    _HEADER = re.compile(
        r"^\s{0,3}#{1,6}\s*",
        flags=re.MULTILINE,
    )

    _EMPHASIS = re.compile(
        r"(\*\*|\*|__|_)",
    )

    _HRULE = re.compile(
        r"^\s*([-*_]){3,}\s*$",
        flags=re.MULTILINE,
    )

    def transform(
        self,
        document: Document,
    ) -> Document:
        """
        Convert Markdown into plain text.
        """

        content = document.content

        #
        # Remove fenced code blocks.
        #
        content = self._CODE_BLOCK.sub(
            "",
            content,
        )

        #
        # Remove inline code markers.
        #
        content = self._INLINE_CODE.sub(
            r"\1",
            content,
        )

        #
        # Remove images.
        #
        content = self._IMAGE.sub(
            "",
            content,
        )

        #
        # Keep only link text.
        #
        content = self._LINK.sub(
            r"\1",
            content,
        )

        #
        # Remove heading markers.
        #
        content = self._HEADER.sub(
            "",
            content,
        )

        #
        # Remove emphasis markers.
        #
        content = self._EMPHASIS.sub(
            "",
            content,
        )

        #
        # Remove horizontal rules.
        #
        content = self._HRULE.sub(
            "",
            content,
        )

        return Document(
            id=document.id,
            content=content,
            metadata=document.metadata,
        )