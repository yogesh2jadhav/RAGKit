from pathlib import Path

from ragkit.loaders.text_loader import TextLoader
from ragkit.models.source_document import SourceDocument


def test_text_loader_supports_txt():

    source = SourceDocument(
        uri="docs/test.txt",
        mime_type="text/plain",
    )

    assert TextLoader.supports(source)


def test_text_loader_does_not_support_pdf():

    source = SourceDocument(
        uri="docs/test.pdf",
        mime_type="application/pdf",
    )

    assert not TextLoader.supports(source)