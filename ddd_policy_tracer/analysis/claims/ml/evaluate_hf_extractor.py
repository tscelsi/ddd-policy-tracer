"""Evaluate a Hugging Face claim extractor against a silver claims dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ddd_policy_tracer.analysis.chunks.chunking_models import DocumentChunk
from ddd_policy_tracer.analysis.claims import (
    HuggingFaceClaimExtractor,
    HuggingFaceClaimExtractorConfig,
)


def run(
    *,
    dataset_path: Path,
    model_name: str,
    max_rows: int | None,
    summary_output_path: Path | None,
) -> dict[str, object]:
    """Score exact span precision/recall/F1 for HF claim extraction."""
    records = _load_records(dataset_path=dataset_path, max_rows=max_rows)
    extractor = HuggingFaceClaimExtractor(
        config=HuggingFaceClaimExtractorConfig(model_name=model_name),
    )

    tp = 0
    fp = 0
    fn = 0
    evaluated_rows = 0

    for record in records:
        chunk_id = str(record.get("chunk_id", ""))
        chunk_text = str(record.get("chunk_text", ""))
        if not chunk_id or not chunk_text:
            continue

        chunk = DocumentChunk(
            chunk_id=chunk_id,
            source_id=str(record.get("source_id", "evaluation")),
            source_document_id=str(record.get("source_document_id", "evaluation")),
            document_checksum=str(record.get("document_checksum", "evaluation")),
            chunk_index=0,
            start_char=0,
            end_char=len(chunk_text),
            chunk_text=chunk_text,
        )
        predicted_spans = {
            (claim.start_char, claim.end_char) for claim in extractor.extract(chunk=chunk)
        }
        gold_spans = set(_record_claim_spans(record=record))

        tp += len(predicted_spans.intersection(gold_spans))
        fp += len(predicted_spans - gold_spans)
        fn += len(gold_spans - predicted_spans)
        evaluated_rows += 1

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    summary: dict[str, object] = {
        "dataset_path": str(dataset_path),
        "model_name": model_name,
        "rows_loaded": len(records),
        "rows_evaluated": evaluated_rows,
        "metrics": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }

    if summary_output_path is not None:
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_output_path.write_text(
            json.dumps(summary, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    return summary


def _record_claim_spans(*, record: dict[str, object]) -> list[tuple[int, int]]:
    """Extract valid claim spans from one span-based silver row."""
    raw_claims = record.get("silver_claims")
    if not isinstance(raw_claims, list):
        return []
    chunk_text = str(record.get("chunk_text", ""))
    spans: list[tuple[int, int]] = []
    for claim in raw_claims:
        if not isinstance(claim, dict):
            continue
        start_char = claim.get("start_char")
        end_char = claim.get("end_char")
        if not isinstance(start_char, int) or not isinstance(end_char, int):
            continue
        if start_char < 0 or end_char <= start_char or end_char > len(chunk_text):
            continue
        spans.append((start_char, end_char))
    return spans


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return zero when denominator is zero for metric safety."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _load_records(*, dataset_path: Path, max_rows: int | None) -> list[dict[str, object]]:
    """Load JSONL dataset rows with optional max row cap."""
    if not dataset_path.exists():
        raise ValueError("dataset_path does not exist")
    records: list[dict[str, object]] = []
    for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        if isinstance(payload, dict):
            records.append(payload)
        if max_rows is not None and len(records) >= max_rows:
            break
    return records


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for HF claim extractor evaluation."""
    parser = argparse.ArgumentParser(
        prog="claims-eval-hf-extractor",
        description="Evaluate HF claim extractor against silver claims JSONL.",
    )
    parser.add_argument("--dataset-path", required=True, help="Claims silver JSONL path")
    parser.add_argument(
        "--model-name",
        default="Babelscape/t5-base-summarization-claim-extractor",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on evaluated rows",
    )
    parser.add_argument(
        "--summary-output-path",
        default=None,
        help="Optional JSON summary output path",
    )
    return parser


def main() -> int:
    """Run HF extractor evaluation from CLI arguments."""
    parser = _build_parser()
    args = parser.parse_args()
    summary = run(
        dataset_path=Path(args.dataset_path),
        model_name=str(args.model_name),
        max_rows=int(args.max_rows) if args.max_rows is not None else None,
        summary_output_path=Path(args.summary_output_path) if args.summary_output_path else None,
    )
    sys.stdout.write(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
