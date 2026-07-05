from __future__ import annotations

from abc import ABC, abstractmethod

from ragkit.models.document import Document
from ragkit.models.source_document import SourceDocument

"""
 => This is like interface of Loader. Other Class have to implement load method.
    It have two methods, 1. supports 2. load
"""
class Loader(ABC):
    """
    Base class for all document loaders.
    """

    @classmethod
    @abstractmethod
    def supports(cls, source: SourceDocument) -> bool:
        """
        Returns True if this loader can process the source document.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, source: SourceDocument) -> Document:
        """
        Load a SourceDocument into a Document.
        """
        raise NotImplementedError