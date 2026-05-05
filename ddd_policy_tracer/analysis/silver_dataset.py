"""Contracts and deterministic utilities for silver label datasets."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, cast

SilverLabelerKind = Literal["llm", "rule", "human"]
SilverEntityType = Literal["POLICY", "ORG", "PERSON", "JURISDICTION", "PROGRAM"]
_SILVER_ENTITY_TYPES: tuple[SilverEntityType, ...] = (
    "POLICY",
    "ORG",
    "PERSON",
    "JURISDICTION",
    "PROGRAM",
)

def validate_claim_silver_record(record: Mapping[str, object]) -> None:
    """Validate one claims silver record contract with lineage metadata."""
    _validate_common_record_fields(record)
    _validate_lineage_metadata(record)
    labels = _require_list(record, "silver_claims")
    for index, label in enumerate(labels):
        if not isinstance(label, Mapping):
            raise ValueError(f"silver_claims[{index}] must be an object")
        evidence_text = _require_str(label, "evidence_text")
        normalized = _require_str(label, "normalized_claim_text")
        if not evidence_text.strip():
            raise ValueError(f"silver_claims[{index}].evidence_text must not be empty")
        if not normalized.strip():
            raise ValueError(
                f"silver_claims[{index}].normalized_claim_text must not be empty",
            )


def validate_entity_silver_record(record: Mapping[str, object]) -> None:
    """Validate one entities silver record contract with lineage metadata."""
    _validate_common_record_fields(record)
    _validate_lineage_metadata(record)
    labels = _require_list(record, "silver_entities")
    for index, label in enumerate(labels):
        if not isinstance(label, Mapping):
            raise ValueError(f"silver_entities[{index}] must be an object")

        start_char = _require_int(label, "start_char")
        end_char = _require_int(label, "end_char")
        if start_char < 0:
            raise ValueError(f"silver_entities[{index}].start_char must be >= 0")
        if end_char <= start_char:
            raise ValueError(
                f"silver_entities[{index}].end_char must be greater than start_char",
            )

        mention_text = _require_str(label, "mention_text")
        if not mention_text.strip():
            raise ValueError(f"silver_entities[{index}].mention_text must not be empty")

        entity_type = _require_str(label, "entity_type")
        if entity_type not in _SILVER_ENTITY_TYPES:
            raise ValueError(
                f"silver_entities[{index}].entity_type must be one of {_SILVER_ENTITY_TYPES}",
            )


def deterministic_sample_records[T](
    records: Sequence[T],
    *,
    sample_size: int,
    seed: int,
) -> list[T]:
    """Return a reproducible sample of records for a fixed seed."""
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")
    if not records:
        raise ValueError("records must not be empty")

    active_sample_size = min(sample_size, len(records))
    rng = random.Random(seed)  # noqa: S311 - deterministic non-crypto sampling
    return rng.sample(list(records), k=active_sample_size)


def deterministic_split_records[T](
    records: Sequence[T],
    *,
    seed: int,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    record_id_fn: Callable[[T], str] | None = None,
) -> dict[str, list[T]]:
    """Split records into reproducible train/dev/test partitions."""
    if not records:
        raise ValueError("records must not be empty")
    _validate_split_ratios(train_ratio=train_ratio, dev_ratio=dev_ratio, test_ratio=test_ratio)

    id_fn = record_id_fn or _default_record_id
    keyed_records = sorted(
        records,
        key=lambda record: _stable_rank_key(seed=seed, record_id=id_fn(record)),
    )

    train_count, dev_count, test_count = _compute_split_counts(
        record_count=len(keyed_records),
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
    )

    train_end = train_count
    dev_end = train_end + dev_count
    return {
        "train": keyed_records[:train_end],
        "dev": keyed_records[train_end:dev_end],
        "test": keyed_records[dev_end : dev_end + test_count],
    }


def _default_record_id[T](record: T) -> str:
    """Read chunk_id as the default stable identifier for one record."""
    if not isinstance(record, Mapping):
        raise ValueError("record_id_fn is required for non-mapping records")
    raw_chunk_id = record.get("chunk_id")
    if not isinstance(raw_chunk_id, str) or not raw_chunk_id.strip():
        raise ValueError("record must include non-empty chunk_id")
    return raw_chunk_id


def _stable_rank_key(*, seed: int, record_id: str) -> str:
    """Compute a deterministic sort key from seed and record id."""
    digest = hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()
    return digest


def _compute_split_counts(
    *,
    record_count: int,
    train_ratio: float,
    dev_ratio: float,
) -> tuple[int, int, int]:
    """Convert split ratios into counts that sum exactly to record count."""
    train_count = int(record_count * train_ratio)
    dev_count = int(record_count * dev_ratio)
    used_count = train_count + dev_count
    test_count = record_count - used_count
    if test_count < 0:
        raise ValueError("split ratios produce invalid negative test size")
    return train_count, dev_count, test_count


def _validate_split_ratios(*, train_ratio: float, dev_ratio: float, test_ratio: float) -> None:
    """Validate split ratios are non-negative and sum to one."""
    if train_ratio < 0 or dev_ratio < 0 or test_ratio < 0:
        raise ValueError("split ratios must be non-negative")
    ratio_sum = train_ratio + dev_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1.0")


def _validate_common_record_fields(record: Mapping[str, object]) -> None:
    """Validate fields shared by claims and entities silver records."""
    for field_name in (
        "chunk_id",
        "source_id",
        "source_document_id",
        "document_checksum",
        "chunk_text",
    ):
        value = _require_str(record, field_name)
        if not value.strip():
            raise ValueError(f"{field_name} must not be empty")


def _validate_lineage_metadata(record: Mapping[str, object]) -> None:
    """Validate required lineage fields for reproducible dataset provenance."""
    for field_name in (
        "labeling_run_id",
        "labeler_kind",
        "labeler_version",
        "label_prompt_version",
        "dataset_version",
        "labeled_at_utc",
    ):
        value = _require_str(record, field_name)
        if not value.strip():
            raise ValueError(f"{field_name} must not be empty")

    labeler_kind = _require_str(record, "labeler_kind")
    if labeler_kind not in ("llm", "rule", "human"):
        raise ValueError("labeler_kind must be one of ('llm', 'rule', 'human')")


def _require_str(payload: Mapping[str, object], field_name: str) -> str:
    """Read one required string field from a payload mapping."""
    raw_value = payload.get(field_name)
    if not isinstance(raw_value, str):
        raise ValueError(f"{field_name} must be a string")
    return raw_value


def _require_int(payload: Mapping[str, object], field_name: str) -> int:
    """Read one required integer field from a payload mapping."""
    raw_value = payload.get(field_name)
    if not isinstance(raw_value, int):
        raise ValueError(f"{field_name} must be an integer")
    return raw_value


def _require_list(payload: Mapping[str, object], field_name: str) -> list[object]:
    """Read one required list field from a payload mapping."""
    raw_value = payload.get(field_name)
    if not isinstance(raw_value, list):
        raise ValueError(f"{field_name} must be a list")
    return cast(list[object], raw_value)
