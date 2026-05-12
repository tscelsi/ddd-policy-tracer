"""Validation and anomaly collection for Stage 5 graph materialization inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GraphAnomaly:
    """Represent one validation or materialization anomaly record."""

    category: str
    severity: str
    message: str
    source_path: str
    line_number: int | None


@dataclass(frozen=True)
class ValidationReport:
    """Capture validation anomalies discovered for required input artifacts."""

    anomalies: list[GraphAnomaly]

    @property
    def anomaly_count(self) -> int:
        """Return number of captured validation anomalies."""
        return len(self.anomalies)


def validate_required_input_paths(
    *,
    chunks_path: Path,
    claims_path: Path,
    entities_path: Path,
) -> None:
    """Fail fast when required input paths do not exist or are not files."""
    _require_file(path=chunks_path, label="chunks")
    _require_file(path=claims_path, label="claims")
    _require_file(path=entities_path, label="entities")


def collect_input_anomalies(
    *,
    chunks_path: Path,
    claims_path: Path,
    entities_path: Path,
) -> ValidationReport:
    """Collect row-level anomalies for required input artifacts."""
    chunk_ids = _collect_chunk_ids(chunks_path)
    anomalies: list[GraphAnomaly] = []

    anomalies.extend(
        _collect_json_parse_anomalies(path=chunks_path)
        + _collect_json_parse_anomalies(path=claims_path)
        + _collect_json_parse_anomalies(path=entities_path),
    )

    anomalies.extend(_collect_claim_reference_anomalies(path=claims_path, chunk_ids=chunk_ids))
    anomalies.extend(_collect_entity_reference_anomalies(path=entities_path, chunk_ids=chunk_ids))

    return ValidationReport(anomalies=anomalies)


def _require_file(*, path: Path, label: str) -> None:
    """Require one input path to exist and point to a file."""
    if not path.exists():
        raise ValueError(f"{label} input path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} input path is not a file: {path}")


def _collect_chunk_ids(path: Path) -> set[str]:
    """Collect chunk IDs from parseable chunk rows for orphan checks."""
    chunk_ids: set[str] = set()
    for payload, _line_number in _iter_json_object_rows(path):
        chunk_id = payload.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            chunk_ids.add(chunk_id)
    return chunk_ids


def _collect_json_parse_anomalies(*, path: Path) -> list[GraphAnomaly]:
    """Collect malformed JSON line anomalies for one JSONL file."""
    anomalies: list[GraphAnomaly] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            anomalies.append(
                GraphAnomaly(
                    category="json_decode_error",
                    severity="error",
                    message="Invalid JSON row",
                    source_path=str(path),
                    line_number=line_number,
                ),
            )
            continue
        if not isinstance(payload, dict):
            anomalies.append(
                GraphAnomaly(
                    category="json_not_object",
                    severity="error",
                    message="Row payload is not a JSON object",
                    source_path=str(path),
                    line_number=line_number,
                ),
            )
    return anomalies


def _collect_claim_reference_anomalies(*, path: Path, chunk_ids: set[str]) -> list[GraphAnomaly]:
    """Collect claim anomalies related to missing fields and orphan chunk references."""
    anomalies: list[GraphAnomaly] = []
    required_fields = {
        "claim_id",
        "chunk_id",
        "source_id",
        "source_document_id",
        "document_checksum",
        "start_char",
        "end_char",
        "evidence_text",
        "normalized_claim_text",
        "confidence",
        "extractor_version",
    }
    for payload, line_number in _iter_json_object_rows(path):
        missing = sorted(field for field in required_fields if field not in payload)
        if missing:
            anomalies.append(
                GraphAnomaly(
                    category="missing_claim_fields",
                    severity="error",
                    message=f"Missing claim fields: {', '.join(missing)}",
                    source_path=str(path),
                    line_number=line_number,
                ),
            )
            continue

        chunk_id = payload.get("chunk_id")
        if not isinstance(chunk_id, str) or chunk_id not in chunk_ids:
            anomalies.append(
                GraphAnomaly(
                    category="orphan_claim_chunk",
                    severity="error",
                    message="Claim references missing chunk_id",
                    source_path=str(path),
                    line_number=line_number,
                ),
            )
    return anomalies


def _collect_entity_reference_anomalies(*, path: Path, chunk_ids: set[str]) -> list[GraphAnomaly]:
    """Collect entity anomalies related to missing fields and orphan chunk references."""
    anomalies: list[GraphAnomaly] = []
    required_fields = {
        "entity_id",
        "chunk_id",
        "source_id",
        "source_document_id",
        "document_checksum",
        "start_char",
        "end_char",
        "mention_text",
        "normalized_mention_text",
        "entity_type",
        "confidence",
        "extractor_version",
    }
    for payload, line_number in _iter_json_object_rows(path):
        missing = sorted(field for field in required_fields if field not in payload)
        if missing:
            anomalies.append(
                GraphAnomaly(
                    category="missing_entity_fields",
                    severity="error",
                    message=f"Missing entity fields: {', '.join(missing)}",
                    source_path=str(path),
                    line_number=line_number,
                ),
            )
            continue

        chunk_id = payload.get("chunk_id")
        if not isinstance(chunk_id, str) or chunk_id not in chunk_ids:
            anomalies.append(
                GraphAnomaly(
                    category="orphan_entity_chunk",
                    severity="error",
                    message="Entity references missing chunk_id",
                    source_path=str(path),
                    line_number=line_number,
                ),
            )
    return anomalies


def _iter_json_object_rows(path: Path) -> list[tuple[dict[str, object], int]]:
    """Return parsed JSON object rows with their source line numbers."""
    rows: list[tuple[dict[str, object], int]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append((payload, line_number))
    return rows
