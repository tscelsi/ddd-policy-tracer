"""Unit tests for canonicalization run module behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ddd_policy_tracer.analysis.canonicalization.run import run_all, run_claims, run_entities


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write JSONL rows to one artifact path for tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _entity_row(
    *,
    entity_id: str,
    chunk_id: str,
    start_char: int,
    end_char: int,
    mention_text: str,
    normalized_mention_text: str,
) -> dict[str, object]:
    """Build one source entity row for canonicalization fixtures."""
    return {
        "entity_id": entity_id,
        "chunk_id": chunk_id,
        "source_id": "australia_institute",
        "source_document_id": "https://example.org/report-1",
        "document_checksum": "checksum-1",
        "start_char": start_char,
        "end_char": end_char,
        "mention_text": mention_text,
        "normalized_mention_text": normalized_mention_text,
        "entity_type": "ORG",
        "confidence": 0.9,
        "extractor_version": "rules-v1",
        "canonical_entity_key": None,
        "metadata": None,
    }


def _claim_row(
    *,
    claim_id: str,
    chunk_id: str,
    start_char: int,
    end_char: int,
    normalized_claim_text: str,
) -> dict[str, object]:
    """Build one source claim row for canonicalization fixtures."""
    return {
        "claim_id": claim_id,
        "chunk_id": chunk_id,
        "source_id": "australia_institute",
        "source_document_id": "https://example.org/report-1",
        "document_checksum": "checksum-1",
        "start_char": start_char,
        "end_char": end_char,
        "evidence_text": "The Australia Institute supports the policy.",
        "normalized_claim_text": normalized_claim_text,
        "confidence": 0.9,
        "claim_type": "descriptive",
        "extractor_version": "rules-v1",
    }


def test_run_entities_writes_canonical_rows_with_deterministic_key(tmp_path: Path) -> None:
    """Canonicalize entities and persist deterministic canonical key output."""
    entities_path = tmp_path / "entities.jsonl"
    entities_canonical_path = tmp_path / "entities_canonical.jsonl"
    _write_jsonl(
        entities_path,
        [
            _entity_row(
                entity_id="entity-1",
                chunk_id="chunk-1",
                start_char=0,
                end_char=20,
                mention_text="The Australia Institute",
                normalized_mention_text="The Australia Institute",
            ),
            _entity_row(
                entity_id="entity-2",
                chunk_id="chunk-1",
                start_char=25,
                end_char=43,
                mention_text="Australia Institute",
                normalized_mention_text="Australia Institute",
            ),
        ],
    )

    report = run_entities(
        entities_path=entities_path,
        entities_canonical_path=entities_canonical_path,
        entity_canonicalizer_version="entity-v1",
    )

    assert report.stage == "entity"
    rows = [
        json.loads(line)
        for line in entities_canonical_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["canonical_entity_key"] == rows[1]["canonical_entity_key"]
    assert rows[0]["canonical_name"] == "australia institute"
    assert rows[0]["canonicalization"]["stage"] == "entity"
    assert rows[0]["canonicalization"]["canonicalizer_version"] == "entity-v1"


def test_run_claims_links_entities_by_span_then_text_fallback(tmp_path: Path) -> None:
    """Link canonical entities to claims using span and fallback precedence."""
    entities_path = tmp_path / "entities.jsonl"
    entities_canonical_path = tmp_path / "entities_canonical.jsonl"
    claims_path = tmp_path / "claims.jsonl"
    claims_canonical_path = tmp_path / "claims_canonical.jsonl"
    _write_jsonl(
        entities_path,
        [
            _entity_row(
                entity_id="entity-span",
                chunk_id="chunk-1",
                start_char=5,
                end_char=15,
                mention_text="Institute",
                normalized_mention_text="Institute",
            ),
            _entity_row(
                entity_id="entity-fallback",
                chunk_id="chunk-1",
                start_char=100,
                end_char=120,
                mention_text="Policy",
                normalized_mention_text="Policy",
            ),
        ],
    )
    _write_jsonl(
        claims_path,
        [
            _claim_row(
                claim_id="claim-1",
                chunk_id="chunk-1",
                start_char=0,
                end_char=40,
                normalized_claim_text="institute should support policy outcomes",
            ),
        ],
    )
    run_entities(
        entities_path=entities_path,
        entities_canonical_path=entities_canonical_path,
        entity_canonicalizer_version="entity-v1",
    )

    report = run_claims(
        claims_path=claims_path,
        entities_canonical_path=entities_canonical_path,
        claims_canonical_path=claims_canonical_path,
        claim_canonicalizer_version="claim-v1",
        required_entity_canonicalizer_version="entity-v1",
    )

    assert report.stage == "claim"
    rows = [
        json.loads(line)
        for line in claims_canonical_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    linked = rows[0]["linked_entities"]
    assert len(linked) == 2
    methods = {entry["entity_id"]: entry["link_method"] for entry in linked}
    assert methods["entity-span"] == "span_overlap"
    assert methods["entity-fallback"] == "text_match_fallback"


def test_run_claims_fails_when_required_entity_version_mismatch(tmp_path: Path) -> None:
    """Reject claim canonicalization when entity canonical version mismatches."""
    entities_path = tmp_path / "entities.jsonl"
    entities_canonical_path = tmp_path / "entities_canonical.jsonl"
    claims_path = tmp_path / "claims.jsonl"
    claims_canonical_path = tmp_path / "claims_canonical.jsonl"
    _write_jsonl(
        entities_path,
        [
            _entity_row(
                entity_id="entity-1",
                chunk_id="chunk-1",
                start_char=0,
                end_char=8,
                mention_text="Institute",
                normalized_mention_text="Institute",
            ),
        ],
    )
    _write_jsonl(
        claims_path,
        [
            _claim_row(
                claim_id="claim-1",
                chunk_id="chunk-1",
                start_char=0,
                end_char=20,
                normalized_claim_text="institute policy",
            ),
        ],
    )
    run_entities(
        entities_path=entities_path,
        entities_canonical_path=entities_canonical_path,
        entity_canonicalizer_version="entity-v1",
    )

    with pytest.raises(ValueError, match="version mismatch"):
        run_claims(
            claims_path=claims_path,
            entities_canonical_path=entities_canonical_path,
            claims_canonical_path=claims_canonical_path,
            claim_canonicalizer_version="claim-v1",
            required_entity_canonicalizer_version="entity-v2",
        )


def test_run_all_runs_entities_before_claims(tmp_path: Path) -> None:
    """Canonicalize all stages in order and emit both artifacts."""
    entities_path = tmp_path / "entities.jsonl"
    entities_canonical_path = tmp_path / "entities_canonical.jsonl"
    claims_path = tmp_path / "claims.jsonl"
    claims_canonical_path = tmp_path / "claims_canonical.jsonl"
    _write_jsonl(
        entities_path,
        [
            _entity_row(
                entity_id="entity-1",
                chunk_id="chunk-1",
                start_char=0,
                end_char=8,
                mention_text="Institute",
                normalized_mention_text="Institute",
            ),
        ],
    )
    _write_jsonl(
        claims_path,
        [
            _claim_row(
                claim_id="claim-1",
                chunk_id="chunk-1",
                start_char=0,
                end_char=20,
                normalized_claim_text="institute policy",
            ),
        ],
    )

    entity_report, claim_report = run_all(
        entities_path=entities_path,
        entities_canonical_path=entities_canonical_path,
        claims_path=claims_path,
        claims_canonical_path=claims_canonical_path,
        entity_canonicalizer_version="entity-v1",
        claim_canonicalizer_version="claim-v1",
    )

    assert entity_report.stage == "entity"
    assert claim_report.stage == "claim"
    assert entities_canonical_path.exists()
    assert claims_canonical_path.exists()


def test_run_entities_is_deterministic_for_same_input_and_version(tmp_path: Path) -> None:
    """Produce byte-identical canonical entity artifact on deterministic rerun."""
    entities_path = tmp_path / "entities.jsonl"
    entities_canonical_path = tmp_path / "entities_canonical.jsonl"
    _write_jsonl(
        entities_path,
        [
            _entity_row(
                entity_id="entity-1",
                chunk_id="chunk-1",
                start_char=0,
                end_char=8,
                mention_text="Institute",
                normalized_mention_text="Institute",
            ),
        ],
    )

    run_entities(
        entities_path=entities_path,
        entities_canonical_path=entities_canonical_path,
        entity_canonicalizer_version="entity-v1",
    )
    first = entities_canonical_path.read_text(encoding="utf-8")
    run_entities(
        entities_path=entities_path,
        entities_canonical_path=entities_canonical_path,
        entity_canonicalizer_version="entity-v1",
    )
    second = entities_canonical_path.read_text(encoding="utf-8")

    assert first == second


def test_run_claims_fails_when_canonical_entities_missing(tmp_path: Path) -> None:
    """Reject claim canonicalization when canonical entities artifact is missing."""
    claims_path = tmp_path / "claims.jsonl"
    entities_canonical_path = tmp_path / "entities_canonical.jsonl"
    claims_canonical_path = tmp_path / "claims_canonical.jsonl"
    _write_jsonl(
        claims_path,
        [
            _claim_row(
                claim_id="claim-1",
                chunk_id="chunk-1",
                start_char=0,
                end_char=20,
                normalized_claim_text="institute policy",
            ),
        ],
    )

    with pytest.raises(ValueError, match="canonical entities input is missing"):
        run_claims(
            claims_path=claims_path,
            entities_canonical_path=entities_canonical_path,
            claims_canonical_path=claims_canonical_path,
            claim_canonicalizer_version="claim-v1",
            required_entity_canonicalizer_version="entity-v1",
        )


def test_run_claims_fails_when_source_coverage_mismatches(tmp_path: Path) -> None:
    """Reject claim canonicalization when source coverage does not match claims."""
    entities_path = tmp_path / "entities.jsonl"
    entities_canonical_path = tmp_path / "entities_canonical.jsonl"
    claims_path = tmp_path / "claims.jsonl"
    claims_canonical_path = tmp_path / "claims_canonical.jsonl"
    _write_jsonl(
        entities_path,
        [
            _entity_row(
                entity_id="entity-1",
                chunk_id="chunk-1",
                start_char=0,
                end_char=8,
                mention_text="Institute",
                normalized_mention_text="Institute",
            ),
        ],
    )
    _write_jsonl(
        claims_path,
        [
            {
                **_claim_row(
                    claim_id="claim-1",
                    chunk_id="chunk-1",
                    start_char=0,
                    end_char=20,
                    normalized_claim_text="institute policy",
                ),
                "source_id": "lowy_institute",
            },
        ],
    )
    run_entities(
        entities_path=entities_path,
        entities_canonical_path=entities_canonical_path,
        entity_canonicalizer_version="entity-v1",
    )

    with pytest.raises(ValueError, match="source_id coverage mismatch"):
        run_claims(
            claims_path=claims_path,
            entities_canonical_path=entities_canonical_path,
            claims_canonical_path=claims_canonical_path,
            claim_canonicalizer_version="claim-v1",
            required_entity_canonicalizer_version="entity-v1",
        )


def test_run_claims_fails_when_checksum_coverage_mismatches(tmp_path: Path) -> None:
    """Reject claim canonicalization when checksum coverage does not match claims."""
    entities_path = tmp_path / "entities.jsonl"
    entities_canonical_path = tmp_path / "entities_canonical.jsonl"
    claims_path = tmp_path / "claims.jsonl"
    claims_canonical_path = tmp_path / "claims_canonical.jsonl"
    _write_jsonl(
        entities_path,
        [
            _entity_row(
                entity_id="entity-1",
                chunk_id="chunk-1",
                start_char=0,
                end_char=8,
                mention_text="Institute",
                normalized_mention_text="Institute",
            ),
        ],
    )
    _write_jsonl(
        claims_path,
        [
            {
                **_claim_row(
                    claim_id="claim-1",
                    chunk_id="chunk-1",
                    start_char=0,
                    end_char=20,
                    normalized_claim_text="institute policy",
                ),
                "document_checksum": "checksum-2",
            },
        ],
    )
    run_entities(
        entities_path=entities_path,
        entities_canonical_path=entities_canonical_path,
        entity_canonicalizer_version="entity-v1",
    )

    with pytest.raises(ValueError, match="document_checksum coverage mismatch"):
        run_claims(
            claims_path=claims_path,
            entities_canonical_path=entities_canonical_path,
            claims_canonical_path=claims_canonical_path,
            claim_canonicalizer_version="claim-v1",
            required_entity_canonicalizer_version="entity-v1",
        )
