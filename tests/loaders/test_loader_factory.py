from ragkit.loaders.loader_factory import LoaderFactory
from ragkit.loaders.text_loader import TextLoader
from ragkit.models.source_document import SourceDocument


def test_loader_factory_returns_text_loader():

    source = SourceDocument(
        uri="docs/test.txt",
        mime_type="text/plain",
    )

    loader = LoaderFactory.get_loader(source)

    assert isinstance(loader, TextLoader)