#__future__ lets you use features from newer Python versions before they become the default behavior.
# It provides easy way to  define class
from __future__ import annotations

'''
 ABC = Abstract Base Class.
 abstractmethod = A method that must be implemented by child classes.
'''
from abc import ABC, abstractmethod
from collections.abc import Iterable

from ragkit.models.source_document import SourceDocument

""""
 => Source is empty interface. which have only one method called discover.
"""
class Source(ABC):
    """
    Base class for all document sources.
    Every document source (folder, PDF, database, S3, etc.) should inherit from this class.
    """

    @abstractmethod
    def discover(self) -> Iterable[SourceDocument]:
        """
        Discover available documents.
        Find all available documents from this source.
        Every source (Local, S3, Azure Blob, GitHub...)
        must implement the discover() method.
        """
        raise NotImplementedError
        """
        raise :- raise means This method has no implementation here.
            If someone somehow calls it, Python raises an error: NotImplementedError
            The child class is responsible for providing the implementation. 
        """