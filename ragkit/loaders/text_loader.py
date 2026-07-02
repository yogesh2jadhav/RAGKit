from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from ragkit.loaders.loader import Loader
from ragkit.models.document import Document
from ragkit.models.source_document import SourceDocument

"""
 There Loader is empty interface and TextLoader Class have implemented this interface with "load" function
"""
class TextLoader(Loader):
    """
    Loads plain text and markdown documents.
    """

    _SUPPORTED_EXTENSIONS = {
        ".txt",
        ".md",
    }

    @classmethod
    def supports(cls, source: SourceDocument) -> bool:
        extension = Path(source.uri).suffix.lower()
        return extension in cls._SUPPORTED_EXTENSIONS

    def load(self, source: SourceDocument) -> Document:
        path = Path(source.uri)
        # Path.read_text() is simpler and closes the file automatically.
        content = path.read_text(encoding="utf-8")
        return Document(
            id=uuid4(),
            content=content,
            metadata={
                "uri": source.uri,
                "mime_type": source.mime_type,
                "filename": path.name,
            },
        )