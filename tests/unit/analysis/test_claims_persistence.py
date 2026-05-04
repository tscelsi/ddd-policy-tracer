"""Unit tests for filesystem claim candidate persistence behavior."""

from __future__ import annotations

from pathlib import Path

from ddd_policy_tracer.analysis.claims import (
    ClaimCandidate,
    FilesystemClaimRepository,
)


def _sample_claim(*, claim_id: str = "claim_1") -> ClaimCandidate:
    """Build one representative claim candidate fixture record."""
    return ClaimCandidate(
        claim_id=claim_id,
        chunk_id="chunk_123",
        source_id="australia_institute",
        source_document_id="https://example.org/report-1",
        document_checksum="checksum-1",
        start_char=10,
        end_char=42,
        evidence_text="Policy settings should reduce emissions by 4.9%.",
        normalized_claim_text="policy settings should reduce emissions by 4.9%.",
        confidence=0.92,
        claim_type=None,
        extractor_version="heuristics-v1",
    )


def test_filesystem_claim_repository_round_trips_claim_records(
    tmp_path: Path,
) -> None:
    """Persist claims to JSONL and load matching records back."""
    repository = FilesystemClaimRepository(tmp_path / "claims.jsonl")
    claims = [_sample_claim(claim_id="claim_1"), _sample_claim(claim_id="claim_2")]

    inserted = repository.add_claims(claims)

    assert inserted == 2
    assert repository.list_claims() == claims


def test_filesystem_claim_repository_filters_by_chunk_id(
    tmp_path: Path,
) -> None:
    """Filter loaded claims by chunk identity when requested."""
    repository = FilesystemClaimRepository(tmp_path / "claims.jsonl")
    claim_a = _sample_claim(claim_id="claim_a")
    claim_b = ClaimCandidate(
        claim_id="claim_b",
        chunk_id="chunk_other",
        source_id=claim_a.source_id,
        source_document_id=claim_a.source_document_id,
        document_checksum=claim_a.document_checksum,
        start_char=claim_a.start_char,
        end_char=claim_a.end_char,
        evidence_text=claim_a.evidence_text,
        normalized_claim_text=claim_a.normalized_claim_text,
        confidence=claim_a.confidence,
        claim_type=claim_a.claim_type,
        extractor_version=claim_a.extractor_version,
    )
    repository.add_claims([claim_a, claim_b])

    assert repository.list_claims(chunk_id="chunk_123") == [claim_a]


def test_filesystem_claim_repository_persists_claim_records_only(
    tmp_path: Path,
) -> None:
    """Write only claim-shaped records into append-only JSONL store."""
    state_path = tmp_path / "claims.jsonl"
    repository = FilesystemClaimRepository(state_path)
    repository.add_claims([_sample_claim()])

    raw_lines = state_path.read_text(encoding="utf-8").splitlines()

    assert len(raw_lines) == 1
    assert '"claim_id"' in raw_lines[0]
    assert '"chunk_id"' in raw_lines[0]
    assert '"status"' not in raw_lines[0]
    assert '"processed_sentences"' not in raw_lines[0]


def test_filesystem_claim_repository_skips_duplicate_claim_records(
    tmp_path: Path,
) -> None:
    """Skip inserting duplicate claims using configured idempotency fields."""
    repository = FilesystemClaimRepository(tmp_path / "claims.jsonl")
    claim = _sample_claim(claim_id="claim_a")

    inserted_first = repository.add_claims([claim])
    inserted_second = repository.add_claims([claim])

    assert inserted_first == 1
    assert inserted_second == 0
    assert repository.list_claims() == [claim]


def test_filesystem_claim_repository_allows_distinct_extractor_versions(
    tmp_path: Path,
) -> None:
    """Persist claims with same span when extractor version differs."""
    repository = FilesystemClaimRepository(tmp_path / "claims.jsonl")
    claim_v1 = _sample_claim(claim_id="claim_v1")
    claim_v2 = ClaimCandidate(
        claim_id="claim_v2",
        chunk_id=claim_v1.chunk_id,
        source_id=claim_v1.source_id,
        source_document_id=claim_v1.source_document_id,
        document_checksum=claim_v1.document_checksum,
        start_char=claim_v1.start_char,
        end_char=claim_v1.end_char,
        evidence_text=claim_v1.evidence_text,
        normalized_claim_text=claim_v1.normalized_claim_text,
        confidence=claim_v1.confidence,
        claim_type=claim_v1.claim_type,
        extractor_version="heuristics-v2",
    )

    inserted = repository.add_claims([claim_v1, claim_v2])

    assert inserted == 2
    assert repository.list_claims() == [claim_v1, claim_v2]
