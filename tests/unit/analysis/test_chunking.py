"""Unit tests for deterministic analysis chunking behavior."""

from ddd_policy_tracer.analysis import (
    ChunkingConfig,
    chunk_document_version,
)
from ddd_policy_tracer.discovery.domain import SourceDocumentVersion


def _sample_version(
    *, text: str, checksum: str = "abc123",
) -> SourceDocumentVersion:
    """Build a representative source document version test fixture."""
    return SourceDocumentVersion(
        source_id="australia_institute",
        source_document_id="https://australiainstitute.org.au/report/sample",
        source_url="https://australiainstitute.org.au/report/sample/",
        published_at="2024-05-01T00:00:00+00:00",
        retrieved_at="2026-04-30T00:00:00+00:00",
        checksum=checksum,
        normalized_text=text,
        raw_content_ref="/tmp/sample.bin",
        content_type="application/pdf",
        created_at="2026-04-30T00:00:00+00:00",
        updated_at="2026-04-30T00:00:00+00:00",
    )


def test_chunk_document_version_returns_empty_for_blank_text() -> None:
    """Return no chunks when version text is blank or whitespace-only."""
    version = _sample_version(text="   ")

    chunks = chunk_document_version(version=version)

    assert chunks == []


def test_chunk_document_version_splits_text_with_overlap() -> None:
    """Split text into stable overlapping chunks with expected offsets."""
    version = _sample_version(text="abcdefghijklmnopqrstuvwxyz")
    config = ChunkingConfig(chunk_size_chars=10, chunk_overlap_chars=3)

    chunks = chunk_document_version(version=version, config=config)

    assert len(chunks) == 4
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3]
    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == [
        (0, 10),
        (7, 17),
        (14, 24),
        (21, 26),
    ]
    assert [chunk.chunk_text for chunk in chunks] == [
        "abcdefghij",
        "hijklmnopq",
        "opqrstuvwx",
        "vwxyz",
    ]


def test_chunk_document_version_is_deterministic_for_same_input() -> None:
    """Produce identical chunk ids and offsets for repeated runs."""
    version = _sample_version(text="A" * 40, checksum="same-checksum")
    config = ChunkingConfig(chunk_size_chars=12, chunk_overlap_chars=2)

    first = chunk_document_version(version=version, config=config)
    second = chunk_document_version(version=version, config=config)

    assert first == second
    assert [chunk.chunk_id for chunk in first] == [
        chunk.chunk_id for chunk in second
    ]


def test_chunk_document_version_binds_chunks_to_document_checksum() -> None:
    """Change chunk identities when the source document checksum changes."""
    text = "Stable text across runs"
    version_a = _sample_version(text=text, checksum="checksum-a")
    version_b = _sample_version(text=text, checksum="checksum-b")
    config = ChunkingConfig(chunk_size_chars=8, chunk_overlap_chars=2)

    chunks_a = chunk_document_version(version=version_a, config=config)
    chunks_b = chunk_document_version(version=version_b, config=config)

    assert [chunk.document_checksum for chunk in chunks_a] == [
        "checksum-a",
    ] * len(chunks_a)
    assert [chunk.document_checksum for chunk in chunks_b] == [
        "checksum-b",
    ] * len(chunks_b)
    assert [chunk.chunk_id for chunk in chunks_a] != [
        chunk.chunk_id for chunk in chunks_b
    ]


def test_chunk_document_version_rejects_invalid_chunking_config() -> None:
    """Raise clear errors for chunk size and overlap misconfiguration."""
    version = _sample_version(text="valid text")

    try:
        chunk_document_version(
            version=version,
            config=ChunkingConfig(chunk_size_chars=0, chunk_overlap_chars=0),
        )
    except ValueError as exc:
        assert str(exc) == "chunk_size_chars must be greater than zero"
    else:
        raise AssertionError("Expected ValueError for zero chunk size")

    try:
        chunk_document_version(
            version=version,
            config=ChunkingConfig(chunk_size_chars=10, chunk_overlap_chars=-1),
        )
    except ValueError as exc:
        assert str(exc) == "chunk_overlap_chars must be zero or greater"
    else:
        raise AssertionError("Expected ValueError for negative overlap")

    try:
        chunk_document_version(
            version=version,
            config=ChunkingConfig(chunk_size_chars=10, chunk_overlap_chars=10),
        )
    except ValueError as exc:
        assert (
            str(exc) == "chunk_overlap_chars must be less than chunk_size_chars"
        )
    else:
        raise AssertionError("Expected ValueError for overlap >= chunk size")
