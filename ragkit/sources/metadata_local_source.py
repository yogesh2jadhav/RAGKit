"""
Purpose
-------
Discovers supported documents from a local directory and
attaches metadata to each discovered document.

Responsibilities
----------------
- Discover supported documents.
- Determine MIME types.
- Attach metadata.
- Yield SourceDocument objects.

Does NOT
--------
- Load documents.
- Read document contents.
"""

from __future__ import annotations

import mimetypes
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ragkit.models.source_document import SourceDocument
from ragkit.sources.source import Source


class MetadataLocalSource(Source):
    """
    Discovers supported documents from a local directory
    and attaches metadata.
    """

    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".md",
    }

    def __init__(
        self,
        directory: str | Path,
        metadata: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        directory
            Directory containing documents.

        metadata
            Mapping of filename -> metadata.

            Example
            -------
            {
                "spark_beginner.txt": {
                    "category": "spark",
                    "level": "beginner",
                }
            }
        """

        self._directory = Path(directory)
        self._metadata = metadata or {}

    def discover(
        self,
    ) -> Iterable[SourceDocument]:
        """
        Discover supported documents and attach metadata.
        """

        if not self._directory.exists():
            raise FileNotFoundError(
                f"Directory '{self._directory}' does not exist."
            )

        if not self._directory.is_dir():
            raise NotADirectoryError(
                f"'{self._directory}' is not a directory."
            )

        for file in self._directory.iterdir():

            if not file.is_file():
                continue

            if file.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            mime_type, _ = mimetypes.guess_type(file)

            yield SourceDocument(
                uri=str(file),
                mime_type=mime_type or "application/octet-stream",
                metadata=dict(
                    self._metadata.get(
                        file.name,
                        {},
                    )
                ),
            )