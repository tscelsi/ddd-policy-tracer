"""Unit tests for silver dataset schema and split utilities."""

from __future__ import annotations

import pytest

from ddd_policy_tracer.analysis.silver_dataset import (
    deterministic_sample_records,
    deterministic_split_records,
    validate_claim_silver_record,
    validate_entity_silver_record,
)


def _base_lineage_fields() -> dict[str, object]:
    """Build required lineage metadata for one silver record fixture."""
    return {
        "labeling_run_id": "run_20260505_01",
        "labeler_kind": "llm",
        "labeler_version": "gpt-4.1-mini",
        "label_prompt_version": "claims-prompt-v1",
        "dataset_version": "claims-silver-v1",
        "labeled_at_utc": "2026-05-05T12:34:56Z",
    }


def _base_chunk_fields(*, chunk_id: str = "chunk_1") -> dict[str, object]:
    """Build required chunk identity fields for one silver record fixture."""
    return {
        "chunk_id": chunk_id,
        "source_id": "australia_institute",
        "source_document_id": "https://example.org/report-1",
        "document_checksum": "checksum-1",
        "chunk_text": "Policy text in one chunk.",
    }


def test_validate_claim_silver_record_accepts_valid_record() -> None:
    """Accept claims silver records that satisfy contract and lineage fields."""
    record = {
        **_base_chunk_fields(),
        **_base_lineage_fields(),
        "silver_claims": [
            {
                "evidence_text": "Policy text",
                "normalized_claim_text": "policy text",
            },
        ],
    }

    validate_claim_silver_record(record)


def test_validate_claim_silver_record_requires_lineage_fields() -> None:
    """Reject claims records that omit required lineage metadata."""
    record = {
        **_base_chunk_fields(),
        "silver_claims": [
            {
                "evidence_text": "Policy text",
                "normalized_claim_text": "policy text",
            },
        ],
    }

    with pytest.raises(ValueError, match="labeling_run_id must be a string"):
        validate_claim_silver_record(record)


def test_validate_entity_silver_record_accepts_valid_record() -> None:
    """Accept entities silver records that satisfy strict span and type checks."""
    record = {
        **_base_chunk_fields(),
        **_base_lineage_fields(),
        "silver_entities": [
            {
                "start_char": 0,
                "end_char": 6,
                "entity_type": "POLICY",
                "mention_text": "Policy",
            },
        ],
    }

    validate_entity_silver_record(record)


def test_validate_entity_silver_record_rejects_invalid_entity_type() -> None:
    """Reject entities records with non-v1 entity type values."""
    record = {
        **_base_chunk_fields(),
        **_base_lineage_fields(),
        "silver_entities": [
            {
                "start_char": 0,
                "end_char": 6,
                "entity_type": "INVALID",
                "mention_text": "Policy",
            },
        ],
    }

    with pytest.raises(ValueError, match="entity_type must be one of"):
        validate_entity_silver_record(record)


def test_deterministic_sample_records_is_reproducible_for_fixed_seed() -> None:
    """Return stable sampled records for a given seed and sample size."""
    records = [{"chunk_id": f"chunk_{idx}"} for idx in range(1, 7)]

    sampled_a = deterministic_sample_records(records, sample_size=3, seed=42)
    sampled_b = deterministic_sample_records(records, sample_size=3, seed=42)
    sampled_c = deterministic_sample_records(records, sample_size=3, seed=99)

    assert sampled_a == sampled_b
    assert sampled_a != sampled_c


def test_deterministic_split_records_is_reproducible_and_complete() -> None:
    """Produce stable train/dev/test splits that cover each record exactly once."""
    records = [{"chunk_id": f"chunk_{idx}"} for idx in range(1, 11)]

    split_a = deterministic_split_records(records, seed=42)
    split_b = deterministic_split_records(records, seed=42)
    split_c = deterministic_split_records(records, seed=99)

    assert split_a == split_b
    assert split_a != split_c

    combined_ids = {
        record["chunk_id"]
        for split_name in ("train", "dev", "test")
        for record in split_a[split_name]
    }
    assert combined_ids == {record["chunk_id"] for record in records}
    assert len(split_a["train"]) == 8
    assert len(split_a["dev"]) == 1
    assert len(split_a["test"]) == 1


def test_deterministic_split_records_validates_ratio_sum() -> None:
    """Reject split configurations whose ratios do not sum to one."""
    records = [{"chunk_id": "chunk_1"}]

    with pytest.raises(ValueError, match=r"split ratios must sum to 1\.0"):
        deterministic_split_records(
            records,
            seed=42,
            train_ratio=0.7,
            dev_ratio=0.2,
            test_ratio=0.2,
        )
