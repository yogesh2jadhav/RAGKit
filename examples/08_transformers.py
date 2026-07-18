"""
Example 08
----------

Transform documents before indexing.

Concepts
--------
- Transformer
- MarkdownTransformer
- WhitespaceTransformer

Run
---
python examples/08_transformers.py

Previous
--------
07_document_indexer.py

Next
----
09_reranker.py
"""

from pathlib import Path
from uuid import uuid4

from ragkit.models.document import Document
from ragkit.transformers.markdown_transformer import MarkdownTransformer
from ragkit.transformers.whitespace_transformer import WhitespaceTransformer

EXAMPLES_DIR = Path(__file__).parent
DOCS_DIR = EXAMPLES_DIR / "docs"


def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:

    print("=" * 70)
    print("Example 08 - Transformers")
    print("=" * 70)

    markdown_file = DOCS_DIR / "spark.md"

    if not markdown_file.exists():
        raise FileNotFoundError(
            f"Example file not found: {markdown_file}"
        )

    #
    # Read the markdown document.
    #
    original_text = markdown_file.read_text(
        encoding="utf-8",
    )

    document = Document(
        id=uuid4(),
        content=original_text,
        metadata={},
    )

    print_section("Original Markdown")

    print(document.content)

    #
    # Apply MarkdownTransformer.
    #
    markdown_transformer = MarkdownTransformer()

    markdown_document = markdown_transformer.transform(
        document,
    )

    print_section("After MarkdownTransformer")

    print(markdown_document.content)

    #
    # Apply WhitespaceTransformer.
    #
    whitespace_transformer = WhitespaceTransformer()

    clean_document = whitespace_transformer.transform(
        markdown_document,
    )

    print_section("After WhitespaceTransformer")

    print(clean_document.content)

    #
    # Summary
    #
    print_section("Summary")

    print("Original Characters :", len(document.content))
    print("Clean Characters    :", len(clean_document.content))

    #
    # Learning recap.
    #
    print_section("What did we learn?")

    print("✓ A Transformer modifies a document before chunking.")
    print("✓ Multiple transformers can be chained together.")
    print("✓ MarkdownTransformer removes Markdown syntax.")
    print("✓ WhitespaceTransformer normalizes formatting.")
    print("✓ Cleaner documents usually produce better embeddings.")

    #
    # Knowledge Check.
    #
    print_section("Knowledge Check")

    print("Q1. Why transform documents before chunking?")
    print()
    print("Answer:")
    print("To remove formatting and keep only meaningful content.")
    print()

    print("Q2. Can multiple transformers be applied?")
    print()
    print("Answer:")
    print("Yes. The output of one transformer can be passed")
    print("to another transformer.")
    print()

    print("Q3. Do transformers generate embeddings?")
    print()
    print("Answer:")
    print("No. They only prepare documents for later stages.")
    print()

    print("=" * 70)
    print("Next Example")
    print("=" * 70)
    print("09_reranker.py")


if __name__ == "__main__":
    main()