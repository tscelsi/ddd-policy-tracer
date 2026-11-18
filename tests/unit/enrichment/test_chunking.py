"""Unit tests for deterministic analysis chunking behavior."""

from dataclasses import replace

import pytest

from ddd_policy_tracer.enrichment.chunking.chunking import MarkdownChunker, SpacyChunker
from ddd_policy_tracer.enrichment.chunking.models import ChunkingConfig
from ddd_policy_tracer.ingestion.domain import SourceDocument


def _sample_version(
    *,
    text: str,
    checksum: str = "abc123",
) -> SourceDocument:
    """Build a representative source document version test fixture."""
    return SourceDocument(
        source_id="australia_institute",
        source_document_id="https://australiainstitute.org.au/report/sample",
        source_url="https://australiainstitute.org.au/report/sample/",
        published_at="2024-05-01T00:00:00+00:00",
        retrieved_at="2026-04-30T00:00:00+00:00",
        checksum=checksum,
        normalized_text=text,
        raw_content_ref="sample.bin",
        content_type="application/pdf",
        created_at="2026-04-30T00:00:00+00:00",
        updated_at="2026-04-30T00:00:00+00:00",
    )


@pytest.fixture
def chunker() -> SpacyChunker:
    """Build one SpacyChunker with default chunking settings for testing."""
    return SpacyChunker()


def test_chunk_document_version_returns_empty_for_blank_text(chunker: SpacyChunker) -> None:
    """Return no chunks when version text is blank or whitespace-only."""
    version = _sample_version(text="   ")

    chunks = chunker.chunk_document(doc=version)

    assert chunks == []


def test_chunk_document_version_groups_sentences_with_overlap(chunker: SpacyChunker) -> None:
    """Group sentences into character-budgeted chunks with sentence overlap."""
    version = _sample_version(
        text=("One short sentence. Two short sentence. Three short sentence. Four short sentence."),
    )
    config = ChunkingConfig(chunk_size_chars=45, chunk_overlap_chars=25)
    chunker.set_config(config)
    chunks = chunker.chunk_document(doc=version)

    assert len(chunks) == 3
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == [
        (0, 39),
        (20, 61),
        (40, 82),
    ]
    assert [chunk.chunk_text for chunk in chunks] == [
        "One short sentence. Two short sentence.",
        "Two short sentence. Three short sentence.",
        "Three short sentence. Four short sentence.",
    ]


def test_chunk_document_version_keeps_long_sentence_as_one_chunk(chunker: SpacyChunker) -> None:
    """A single long sentence remains intact even when above the char budget."""
    long_sentence = "A" * 80 + "."
    version = _sample_version(text=long_sentence)
    config = ChunkingConfig(chunk_size_chars=20, chunk_overlap_chars=5)
    chunker.set_config(config)
    chunks = chunker.chunk_document(doc=version)

    assert len(chunks) == 1
    assert chunks[0].chunk_text == long_sentence
    assert (chunks[0].start_char, chunks[0].end_char) == (0, len(long_sentence))


def test_chunk_document_version_is_deterministic_for_same_input(chunker: SpacyChunker) -> None:
    """Produce identical chunk ids and offsets for repeated runs."""
    version = _sample_version(
        text="Alpha sentence. Beta sentence. Gamma sentence.",
        checksum="same-checksum",
    )
    config = ChunkingConfig(chunk_size_chars=30, chunk_overlap_chars=15)
    chunker.set_config(config)
    first = chunker.chunk_document(doc=version)
    second = chunker.chunk_document(doc=version)

    assert first == second
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_chunk_document_version_binds_chunks_to_document_checksum(chunker: SpacyChunker) -> None:
    """Change chunk identities when the source document checksum changes."""
    text = "Stable sentence. Another stable sentence."
    version_a = _sample_version(text=text, checksum="checksum-a")
    version_b = _sample_version(text=text, checksum="checksum-b")
    config = ChunkingConfig(chunk_size_chars=25, chunk_overlap_chars=12)
    chunker.set_config(config)
    chunks_a = chunker.chunk_document(doc=version_a)
    chunks_b = chunker.chunk_document(doc=version_b)

    assert [chunk.document_checksum for chunk in chunks_a] == [
        "checksum-a",
    ] * len(chunks_a)
    assert [chunk.document_checksum for chunk in chunks_b] == [
        "checksum-b",
    ] * len(chunks_b)
    assert [chunk.chunk_id for chunk in chunks_a] != [chunk.chunk_id for chunk in chunks_b]


def test_chunk_document_version_rejects_invalid_chunking_config(chunker: SpacyChunker) -> None:
    """Raise clear errors for chunk size and overlap misconfiguration."""
    version = _sample_version(text="valid text")

    chunker.set_config(ChunkingConfig(chunk_size_chars=0, chunk_overlap_chars=0))
    with pytest.raises(ValueError, match="chunk_size_chars must be greater than zero"):
        chunker.chunk_document(doc=version)

    chunker.set_config(ChunkingConfig(chunk_size_chars=10, chunk_overlap_chars=-1))
    with pytest.raises(ValueError, match="chunk_overlap_chars must be zero or greater"):
        chunker.chunk_document(doc=version)

    chunker.set_config(ChunkingConfig(chunk_size_chars=10, chunk_overlap_chars=10))
    with pytest.raises(ValueError, match="chunk_overlap_chars must be less than chunk_size_chars"):
        chunker.chunk_document(doc=version)


def test_chunkers_surface_hansard_metadata_as_typed_fields() -> None:
    """Keep Hansard search fields explicit while retaining source metadata."""
    version = _sample_version(
        text="A parliamentary speech.",
    )
    version = replace(
        version,
        source_id="hansard",
        metadata={
            "talker": {
                "name": "Member A",
                "party": "Example Party",
                "electorate": "Example Electorate",
            },
            "talker_id": "talker-1",
            "house": "senate",
            "chamber": "main",
            "bill_ids": ["bill-1"],
            "debate_category": "Bills",
            "debate_titles": ["Second reading"],
            "speech_part_type": "speech",
        },
    )

    chunks = MarkdownChunker().chunk_document(doc=version)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.published_at == version.published_at
    assert chunk.speaker_name == "Member A"
    assert chunk.talker_id == "talker-1"
    assert chunk.party == "Example Party"
    assert chunk.electorate == "Example Electorate"
    assert chunk.house == "senate"
    assert chunk.chamber == "main"
    assert chunk.bill_ids == ["bill-1"]
    assert chunk.debate_category == "Bills"
    assert chunk.debate_titles == ["Second reading"]
    assert chunk.speech_part_type == "speech"
    assert chunk.metadata == {"doc_metadata": version.metadata}
    assert chunk.update_embedding([0.1]).talker_id == "talker-1"
