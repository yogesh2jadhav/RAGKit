from __future__ import annotations

from abc import ABC, abstractmethod

from ragkit.models.document import Document
from ragkit.models.source_document import SourceDocument

"""
 This is like empty interface of Loader. Other Class have to implement load method.
"""
class Loader(ABC):
    """
    Base class for all document loaders.
    """

    @abstractmethod
    def load(self, source: SourceDocument) -> Document:
        """
        Load a SourceDocument into a Document.
        """
        raise NotImplementedError