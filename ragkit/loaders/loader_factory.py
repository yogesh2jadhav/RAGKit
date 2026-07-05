from __future__ import annotations

from ragkit.loaders.loader import Loader
from ragkit.loaders.text_loader import TextLoader
from ragkit.models.source_document import SourceDocument

'''
 => LoaderFactory - is a class of loader factory based on extention it will select which load code have to use.
'''
class LoaderFactory:
    '''
    => _LOADERS list of all supported loader as of now we have only Text loader, futur we will have PDF loader
    in that case we will write
    _LOADERS = (TextLoader, PDFLoader,.....)
    '''
    _LOADERS = (TextLoader,)

    # @classmethod : Makes this a class(get_loader) method. You can call it without creating an object.
    # It's like static Class and Method in java.
    @classmethod
    def get_loader(
        cls, #cls is just like self but only for class methods.
        source: SourceDocument,
    ) -> Loader:
        for loader in cls._LOADERS:
            if loader.supports(source): # => if condition match you will get correct loader refreance as return.
                return loader()

        raise ValueError(
            f"No loader found for '{source.uri}'"
        )