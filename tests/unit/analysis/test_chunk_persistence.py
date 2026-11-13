"""Unit tests for chunk persistence adapters and orchestration."""

from pathlib import Path

from ddd_policy_tracer.__DEPRECATED_analysis import (
    ChunkingConfig,
    chunk_and_persist_documents,
)
from ddd_policy_tracer.enrichment.chunking.chunking import SpacyChunker
from ddd_policy_tracer.enrichment.chunking.models import DocumentChunk
from ddd_policy_tracer.ingestion.domain import SourceDocument


def _sample_version(
    *,
    source_document_id: str = "https://example.org/report-1",
    checksum: str = "checksum-1",
    text: str = "A" * 64,
) -> SourceDocument:
    """Build one representative source document version fixture."""
    return SourceDocument(
        source_id="australia_institute",
        source_document_id=source_document_id,
        source_url=f"{source_document_id}/",
        published_at="2024-01-01T00:00:00+00:00",
        retrieved_at="2026-04-30T00:00:00+00:00",
        checksum=checksum,
        normalized_text=text,
        raw_content_ref="/tmp/sample.bin",
        content_type="application/pdf",
        created_at="2026-04-30T00:00:00+00:00",
        updated_at="2026-04-30T00:00:00+00:00",
    )


def _sample_chunk(*, document_checksum: str = "checksum-1") -> DocumentChunk:
    """Build one representative chunk record fixture."""
    return DocumentChunk(
        chunk_id="chunk_abc",
        source_id="australia_institute",
        source_document_id="https://example.org/report-1",
        document_checksum=document_checksum,
        chunk_index=0,
        start_char=0,
        end_char=10,
        chunk_text="abcdefghij",
    )


def test_chunk_and_persist_is_idempotent_for_existing_version(
    tmp_path: Path,
) -> None:
    """Skip persistence when chunks already exist for a document version."""
    state_path = tmp_path / "chunks.db"
    version = _sample_version(text="0123456789" * 10)
    config = ChunkingConfig(chunk_size_chars=20, chunk_overlap_chars=5)
    chunker = SpacyChunker(config=config)
    first_report = chunk_and_persist_documents(
        documents=[version],
        state_path=state_path,
        chunker=chunker,
    )
    second_report = chunk_and_persist_documents(
        documents=[version],
        state_path=state_path,
        chunker=chunker,
    )

    assert first_report.processed_documents == 1
    assert first_report.chunked_documents == 1
    assert first_report.skipped_documents == 0
    assert first_report.persisted_chunks > 0

    assert second_report.processed_documents == 1
    assert second_report.chunked_documents == 0
    assert second_report.skipped_documents == 1
    assert second_report.persisted_chunks == 0


def test_chunk_and_persist_supports_filesystem_backend(
    tmp_path: Path,
) -> None:
    """Persist chunk output into filesystem JSONL backend."""
    state_path = tmp_path / "chunks.jsonl"
    version = _sample_version(text="ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    report = chunk_and_persist_documents(
        documents=[version],
        chunker=SpacyChunker(ChunkingConfig(chunk_size_chars=10, chunk_overlap_chars=2)),
        state_path=state_path,
    )

    assert report.processed_documents == 1
    assert report.chunked_documents == 1
    assert report.skipped_documents == 0
    assert report.persisted_chunks == 1
    assert state_path.exists()
