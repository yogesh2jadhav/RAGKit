from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from ragkit.loaders.loader import Loader
from ragkit.models.document import Document
from ragkit.models.source_document import SourceDocument

"""
 => TextLoader Class have implemented loader interface and it's two methods, 1. supports 2. load
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
        """
        => Returns True if the source document is supported. Based on input Source Document (source_document.py)
        """
        extension = Path(source.uri).suffix.lower()
        return extension in cls._SUPPORTED_EXTENSIONS

    def load(self, source: SourceDocument) -> Document:
        path = Path(source.uri)
        # => Path.read_text() is simpler and closes the file automatically.
        content = path.read_text(encoding="utf-8")
        # => This method create Document object and return it. which following infomation.
        return Document(
            id=uuid4(),
            content=content,
            metadata={
                "uri": source.uri,
                "mime_type": source.mime_type,
                "filename": path.name,
            },
        )