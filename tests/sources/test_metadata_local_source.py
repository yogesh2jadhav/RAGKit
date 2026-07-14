from pathlib import Path

from pytest import raises

from ragkit.sources.metadata_local_source import MetadataLocalSource


def test_discovers_supported_documents(tmp_path):
    """
    Verify supported documents are discovered.
    """

    (tmp_path / "one.txt").write_text(
        "hello",
        encoding="utf-8",
    )

    (tmp_path / "two.md").write_text(
        "hello",
        encoding="utf-8",
    )

    (tmp_path / "three.pdf").write_text(
        "ignored",
        encoding="utf-8",
    )

    source = MetadataLocalSource(
        directory=str(tmp_path),
    )

    documents = list(
        source.discover(),
    )

    assert len(documents) == 2

    names = sorted(
        Path(document.uri).name
        for document in documents
    )

    assert names == [
        "one.txt",
        "two.md",
    ]


def test_metadata_is_attached():
    """
    Verify metadata is attached to matching files.
    """

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:

        path = Path(directory)

        (path / "spark.txt").write_text(
            "Apache Spark",
            encoding="utf-8",
        )

        source = MetadataLocalSource(
            directory=directory,
            metadata={
                "spark.txt": {
                    "category": "spark",
                    "level": "beginner",
                },
            },
        )

        document = next(
            source.discover(),
        )

        assert document.metadata == {
            "category": "spark",
            "level": "beginner",
        }


def test_missing_metadata_returns_empty_dict():
    """
    Verify missing metadata defaults to an empty dictionary.
    """

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:

        path = Path(directory)

        (path / "python.txt").write_text(
            "Python",
            encoding="utf-8",
        )

        source = MetadataLocalSource(
            directory=directory,
        )

        document = next(
            source.discover(),
        )

        assert document.metadata == {}


def test_directory_must_exist():
    """
    Verify missing directories are rejected.
    """

    source = MetadataLocalSource(
        directory="missing_directory",
    )

    with raises(FileNotFoundError):
        list(
            source.discover(),
        )


def test_directory_must_be_directory(tmp_path):
    """
    Verify files are rejected.
    """

    file = tmp_path / "test.txt"

    file.write_text(
        "hello",
        encoding="utf-8",
    )

    source = MetadataLocalSource(
        directory=str(file),
    )

    with raises(NotADirectoryError):
        list(
            source.discover(),
        )


def test_metadata_is_copied():
    """
    Verify metadata is copied before being attached.
    """

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:

        path = Path(directory)

        (path / "spark.txt").write_text(
            "Apache Spark",
            encoding="utf-8",
        )

        metadata = {
            "spark.txt": {
                "category": "spark",
            },
        }

        source = MetadataLocalSource(
            directory=directory,
            metadata=metadata,
        )

        document = next(
            source.discover(),
        )

        metadata["spark.txt"]["category"] = "python"

        assert document.metadata == {
            "category": "spark",
        }