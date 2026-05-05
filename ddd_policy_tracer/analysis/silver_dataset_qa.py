"""QA diagnostics for claims and entities silver dataset artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from ddd_policy_tracer.analysis.silver_dataset import (
    validate_claim_silver_record,
    validate_entity_silver_record,
)


def run(
    *,
    claims_dataset_path: Path,
    entities_dataset_path: Path,
    output_path: Path | None,
) -> dict[str, object]:
    """Generate combined QA diagnostics for claims and entities silver datasets."""
    claims_summary = _qa_claims_dataset(dataset_path=claims_dataset_path)
    entities_summary = _qa_entities_dataset(dataset_path=entities_dataset_path)
    summary: dict[str, object] = {
        "claims": claims_summary,
        "entities": entities_summary,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    return summary


def _qa_claims_dataset(*, dataset_path: Path) -> dict[str, object]:
    """Compute schema and label-shape diagnostics for claims silver rows."""
    rows, parse_failures = _parse_jsonl_rows(dataset_path=dataset_path)
    invalid_rows = 0
    empty_label_rows = 0
    total_claims = 0

    for row in rows:
        try:
            validate_claim_silver_record(row)
        except ValueError:
            invalid_rows += 1
            continue

        labels = row["silver_claims"]
        if not isinstance(labels, list):
            invalid_rows += 1
            continue
        total_claims += len(labels)
        if len(labels) == 0:
            empty_label_rows += 1

    valid_rows = len(rows) - invalid_rows
    empty_label_rate = _safe_divide(empty_label_rows, valid_rows)
    return {
        "dataset_path": str(dataset_path),
        "parsed_rows": len(rows),
        "parse_failures": parse_failures,
        "invalid_rows": invalid_rows,
        "valid_rows": valid_rows,
        "total_claims": total_claims,
        "empty_label_rows": empty_label_rows,
        "empty_label_rate": empty_label_rate,
    }


def _qa_entities_dataset(*, dataset_path: Path) -> dict[str, object]:
    """Compute schema, type, and offset diagnostics for entities silver rows."""
    rows, parse_failures = _parse_jsonl_rows(dataset_path=dataset_path)
    invalid_rows = 0
    empty_label_rows = 0
    total_entities = 0
    invalid_type_mentions = 0
    offset_mismatch_mentions = 0
    type_distribution: Counter[str] = Counter()

    for row in rows:
        try:
            validate_entity_silver_record(row)
        except ValueError:
            invalid_rows += 1
            continue

        labels = row["silver_entities"]
        if not isinstance(labels, list):
            invalid_rows += 1
            continue

        if len(labels) == 0:
            empty_label_rows += 1

        chunk_text = row["chunk_text"]
        if not isinstance(chunk_text, str):
            invalid_rows += 1
            continue

        for label in labels:
            if not isinstance(label, dict):
                invalid_rows += 1
                continue
            entity_type = label.get("entity_type")
            if not isinstance(entity_type, str):
                invalid_type_mentions += 1
                continue

            start_char = label.get("start_char")
            end_char = label.get("end_char")
            mention_text = label.get("mention_text")
            if (
                not isinstance(start_char, int)
                or not isinstance(end_char, int)
                or not isinstance(mention_text, str)
            ):
                offset_mismatch_mentions += 1
                continue

            if start_char < 0 or end_char > len(chunk_text) or end_char <= start_char:
                offset_mismatch_mentions += 1
                continue

            if chunk_text[start_char:end_char] != mention_text:
                offset_mismatch_mentions += 1
                continue

            total_entities += 1
            type_distribution[entity_type] += 1

    valid_rows = len(rows) - invalid_rows
    empty_label_rate = _safe_divide(empty_label_rows, valid_rows)
    return {
        "dataset_path": str(dataset_path),
        "parsed_rows": len(rows),
        "parse_failures": parse_failures,
        "invalid_rows": invalid_rows,
        "valid_rows": valid_rows,
        "total_entities": total_entities,
        "empty_label_rows": empty_label_rows,
        "empty_label_rate": empty_label_rate,
        "invalid_type_mentions": invalid_type_mentions,
        "offset_mismatch_mentions": offset_mismatch_mentions,
        "type_distribution": dict(sorted(type_distribution.items())),
    }


def _parse_jsonl_rows(*, dataset_path: Path) -> tuple[list[dict[str, object]], int]:
    """Parse JSONL rows and return valid objects with parse failure count."""
    if not dataset_path.exists():
        raise ValueError(f"dataset_path does not exist: {dataset_path}")

    rows: list[dict[str, object]] = []
    parse_failures = 0
    for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            parse_failures += 1
            continue
        if not isinstance(payload, dict):
            parse_failures += 1
            continue
        rows.append(payload)
    return rows, parse_failures


def _safe_divide(numerator: int, denominator: int) -> float:
    """Return zero when denominator is zero for stable QA metrics."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for silver dataset QA diagnostics."""
    parser = argparse.ArgumentParser(
        prog="analysis-silver-dataset-qa",
        description="Generate QA diagnostics for claims and entities silver datasets.",
    )
    parser.add_argument("--claims-dataset-path", required=True, help="Path to claims silver JSONL")
    parser.add_argument(
        "--entities-dataset-path",
        required=True,
        help="Path to entities silver JSONL",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional path to write QA summary JSON",
    )
    return parser


def main() -> int:
    """Run silver dataset QA diagnostics from CLI arguments."""
    parser = _build_parser()
    args = parser.parse_args()
    summary = run(
        claims_dataset_path=Path(args.claims_dataset_path),
        entities_dataset_path=Path(args.entities_dataset_path),
        output_path=Path(args.output_path) if args.output_path else None,
    )
    sys.stdout.write(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
