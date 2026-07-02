from __future__ import annotations

from pathlib import Path

from ragkit.loaders.loader import Loader
from ragkit.loaders.text_loader import TextLoader
from ragkit.models.source_document import SourceDocument

'''
 LoaderFactory - is a class of loader factory based on extention it will select which load code have to use.
'''
class LoaderFactory:
    # A dictionary/list that stores supported file types.
    _LOADERS = {
        ".txt": TextLoader,
        ".md": TextLoader,
    }

    # @classmethod : Makes this a class(get_loader) method. You can call it without creating an object.
    # It's like static Class and Method in java.
    @classmethod
    def get_loader(
        cls, #cls is just like self but only for class methods.
        source: SourceDocument,
    ) -> Loader:

        extension = Path(source.uri).suffix.lower()

        loader = cls._LOADERS.get(extension)

        if loader is None:
            raise ValueError(
                f"No loader registered for '{extension}'"
            )

        return loader()