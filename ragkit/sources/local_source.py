from __future__ import annotations

import mimetypes
from collections.abc import Iterable
from pathlib import Path

from ragkit.models.source_document import SourceDocument
from ragkit.sources.source import Source

'''
 Local_Source(Source) here - Source is empty interface. and Local_Source is implemented, method of Source.discover() 
'''
class LocalSource(Source):
    """
    Discovers supported documents from a local directory.
    """

    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".md",
    }

    def __init__(self, directory: str) -> None:
        self._directory = Path(directory)

    def discover(self) -> Iterable[SourceDocument]:
        """
        Discover all supported documents in the directory.
        """

        if not self._directory.exists():
            raise FileNotFoundError(
                f"Directory '{self._directory}' does not exist."
            )

        if not self._directory.is_dir():
            raise NotADirectoryError(
                f"'{self._directory}' is not a directory."
            )
        '''
        for file in self._directory.iterdir(): file will hold actula file object which is read by 
        self._directory.iterdir()
        '''
        for file in self._directory.iterdir():

            if not file.is_file():
                continue

            if file.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            ''' 
            mimetypes is build in method into Python which return => ("text/plain", None) something like this.
            and if not able to find info then => application/octet-stream. 
            '''
            mime_type, _ = mimetypes.guess_type(file)

            yield SourceDocument(
                uri=str(file),
                mime_type=mime_type or "application/octet-stream",
            )