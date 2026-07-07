"""
Example 01
----------

Load documents from a source.

Concepts
--------
- Source
- SourceDocument
- LoaderFactory
- Loader
- Document

Run
---
python examples/01_load_documents.py

Next
----
02_chunk_documents.py
"""

from pathlib import Path

from ragkit.loaders.loader_factory import LoaderFactory
from ragkit.sources.local_source import LocalSource


def main() -> None:
    """
    Demonstrates loading documents.
    """
    EXAMPLES_DIR = Path(__file__).parent
    DOCS_DIR = EXAMPLES_DIR / "docs"
    print("=" * 70)
    print("Example 01 - Load Documents")
    print("=" * 70)

    #
    # Example documents.
    #
    docs_directory = Path(__file__).parent / "docs"

    if not docs_directory.exists():
        raise FileNotFoundError(f"Example documents not found: {docs_directory}")

    #
    # A Source discovers available documents.
    #
    # It does NOT load them.
    #
    source = LocalSource(
        DOCS_DIR,
    )

    document_count = 0

    #
    # Iterate through discovered files.
    #
    for source_document in source.discover():

        print()

        print(f"File      : {source_document.uri}")
        print(f"MIME Type : {source_document.mime_type}")

        #
        # LoaderFactory selects the appropriate
        # loader based on the document type.
        #
        loader = LoaderFactory.get_loader(
            source_document,
        )

        #
        # Load the document.
        #
        document = loader.load(
            source_document,
        )

        document_count += 1

        print(f"Characters: {len(document.content)}")

        preview = document.content[:200]

        if len(document.content) > 200:
            preview += "..."

        print()
        print("Preview")
        print("-" * 70)
        print(preview)
        print("-" * 70)

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"Documents Loaded : {document_count}")

    print()
    print("Next Example")
    print("------------")
    print("02_chunk_documents.py")


if __name__ == "__main__":
    main()