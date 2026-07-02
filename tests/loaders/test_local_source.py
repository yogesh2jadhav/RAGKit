from ragkit.sources.local_source import LocalSource


def test_discover_documents():

    source = LocalSource("docs")

    documents = list(source.discover())

    assert len(documents) > 0