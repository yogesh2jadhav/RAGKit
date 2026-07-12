from pytest import raises

from ragkit.exceptions.loader_error import LoaderError
from ragkit.loaders.loader_factory import LoaderFactory
from ragkit.loaders.text_loader import TextLoader
from ragkit.models.source_document import SourceDocument


def test_loader_factory_returns_text_loader():
    """
    Verify LoaderFactory returns TextLoader
    for a supported text document.
    """

    source = SourceDocument(
        uri="docs/test.txt",
        mime_type="text/plain",
    )

    loader = LoaderFactory.get_loader(source)

    assert isinstance(loader, TextLoader)


def test_loader_factory_raises_loader_error():
    """
    Verify LoaderFactory raises LoaderError
    when no loader supports the document.
    """

    source = SourceDocument(
        uri="docs/test.pdf",
        mime_type="application/pdf",
    )

    with raises(LoaderError):
        LoaderFactory.get_loader(source)