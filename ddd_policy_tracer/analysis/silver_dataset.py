"""Contracts and deterministic utilities for silver label datasets."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator

SilverLabelerKind = Literal["llm", "rule", "human"]
SilverEntityType = Literal["POLICY", "ORG", "PERSON", "JURISDICTION", "PROGRAM"]


class SilverLineage(BaseModel):
    """Represent required lineage metadata for one silver label record."""

    labeling_run_id: str = Field(min_length=1)
    labeler_kind: SilverLabelerKind
    labeler_version: str = Field(min_length=1)
    label_prompt_version: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    labeled_at_utc: str = Field(min_length=1)


class SilverRecordBase(BaseModel):
    """Represent chunk identity and text fields common to silver records."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_document_id: str = Field(min_length=1)
    document_checksum: str = Field(min_length=1)
    chunk_text: str = Field(min_length=1)
    labeling_run_id: str = Field(min_length=1)
    labeler_kind: SilverLabelerKind
    labeler_version: str = Field(min_length=1)
    label_prompt_version: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    labeled_at_utc: str = Field(min_length=1)


class SilverClaimLabel(BaseModel):
    """Represent one silver-labeled claim span in a chunk."""

    model_config = ConfigDict(extra="forbid")

    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)

    @field_validator("end_char")
    @classmethod
    def _validate_end_char(cls, value: int, info: ValidationInfo) -> int:
        """Ensure claim end offset is strictly greater than start offset."""
        start_char = info.data.get("start_char")
        if isinstance(start_char, int) and value <= start_char:
            raise ValueError("end_char must be greater than start_char")
        return value


class ClaimsSilverRecord(SilverRecordBase):
    """Represent one validated claims silver dataset row."""

    silver_claims: list[SilverClaimLabel]


class SilverEntityLabel(BaseModel):
    """Represent one silver-labeled entity mention."""

    model_config = ConfigDict(extra="forbid")

    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    mention_text: str = Field(min_length=1)
    entity_type: SilverEntityType

    @field_validator("end_char")
    @classmethod
    def _validate_end_char(cls, value: int, info: ValidationInfo) -> int:
        """Ensure end offset is strictly greater than start offset."""
        start_char = info.data.get("start_char")
        if isinstance(start_char, int) and value <= start_char:
            raise ValueError("end_char must be greater than start_char")
        return value


class EntitiesSilverRecord(SilverRecordBase):
    """Represent one validated entities silver dataset row."""

    silver_entities: list[SilverEntityLabel]


def validate_claim_silver_record(record: Mapping[str, object]) -> None:
    """Validate one claims silver record contract with lineage metadata."""
    try:
        ClaimsSilverRecord.model_validate(record)
    except ValidationError as exc:
        raise ValueError(_flatten_validation_error(exc)) from exc


def validate_entity_silver_record(record: Mapping[str, object]) -> None:
    """Validate one entities silver record contract with lineage metadata."""
    try:
        EntitiesSilverRecord.model_validate(record)
    except ValidationError as exc:
        raise ValueError(_flatten_validation_error(exc)) from exc


def parse_claim_silver_record(record: Mapping[str, object]) -> ClaimsSilverRecord:
    """Parse and return one typed claims silver record."""
    try:
        return ClaimsSilverRecord.model_validate(record)
    except ValidationError as exc:
        raise ValueError(_flatten_validation_error(exc)) from exc


def parse_entity_silver_record(record: Mapping[str, object]) -> EntitiesSilverRecord:
    """Parse and return one typed entities silver record."""
    try:
        return EntitiesSilverRecord.model_validate(record)
    except ValidationError as exc:
        raise ValueError(_flatten_validation_error(exc)) from exc


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


def _flatten_validation_error(error: ValidationError) -> str:
    """Render one concise error message from Pydantic validation errors."""
    first = error.errors()[0]
    location = ".".join(str(part) for part in first.get("loc", ("record",)))
    message = str(first.get("msg", "validation failed"))
    return f"{location}: {message}"
