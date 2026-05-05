"""Evaluate rule-based entity extractor against a labeled chunk dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ddd_policy_tracer.analysis.chunks.chunking_models import DocumentChunk
from ddd_policy_tracer.analysis.entities.extractors import (
    RuleBasedEntityExtractorConfig,
    RuleBasedSentenceEntityExtractor,
)
from ddd_policy_tracer.analysis.entities.models import EntityType
from ddd_policy_tracer.utils.logger import configure_logging, get_logger

LOGGER = get_logger(__name__, ctx="entities_evaluation")
_ENTITY_TYPES: tuple[EntityType, ...] = (
    "POLICY",
    "ORG",
    "PERSON",
    "JURISDICTION",
    "PROGRAM",
)


def run(
    *,
    dataset_path: Path,
    output_path: Path | None,
    policy_threshold: float,
    org_threshold: float,
    person_threshold: float,
    jurisdiction_threshold: float,
    program_threshold: float,
) -> dict[str, object]:
    """Evaluate extractor precision/recall metrics over labeled entity mentions."""
    records = _load_dataset(dataset_path)
    extractor = RuleBasedSentenceEntityExtractor(
        RuleBasedEntityExtractorConfig(
            policy_threshold=policy_threshold,
            org_threshold=org_threshold,
            person_threshold=person_threshold,
            jurisdiction_threshold=jurisdiction_threshold,
            program_threshold=program_threshold,
        ),
    )
    run_logger = LOGGER.bind(
        dataset_path=str(dataset_path),
        dataset_records=len(records),
        policy_threshold=policy_threshold,
        org_threshold=org_threshold,
        person_threshold=person_threshold,
        jurisdiction_threshold=jurisdiction_threshold,
        program_threshold=program_threshold,
    )
    run_logger.info("evaluation started")

    totals = {"tp": 0, "fp": 0, "fn": 0}
    per_type_totals: dict[EntityType, dict[str, int]] = {
        entity_type: {"tp": 0, "fp": 0, "fn": 0}
        for entity_type in _ENTITY_TYPES
    }
    per_chunk: list[dict[str, object]] = []

    for index, record in enumerate(records, start=1):
        chunk = DocumentChunk(
            chunk_id=record["chunk_id"],
            source_id=record["source_id"],
            source_document_id=record["source_document_id"],
            document_checksum=record["document_checksum"],
            chunk_index=0,
            start_char=0,
            end_char=len(record["chunk_text"]),
            chunk_text=record["chunk_text"],
        )
        predicted = {
            _to_match_key(
                chunk_id=chunk.chunk_id,
                start_char=entity.start_char,
                end_char=entity.end_char,
                entity_type=entity.entity_type,
            )
            for entity in extractor.extract(chunk=chunk)
        }
        gold = {
            _to_match_key(
                chunk_id=chunk.chunk_id,
                start_char=mention["start_char"],
                end_char=mention["end_char"],
                entity_type=mention["entity_type"],
            )
            for mention in record["gold_entities"]
        }

        tp_set = predicted.intersection(gold)
        fp_set = predicted - gold
        fn_set = gold - predicted

        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

        for key in tp_set:
            per_type_totals[key[3]]["tp"] += 1
        for key in fp_set:
            per_type_totals[key[3]]["fp"] += 1
        for key in fn_set:
            per_type_totals[key[3]]["fn"] += 1

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_chunk.append(
            {
                "chunk_id": chunk.chunk_id,
                "predicted_entities": len(predicted),
                "gold_entities": len(gold),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
        )
        run_logger.debug(
            "chunk evaluated record_index=%s chunk_id=%s predicted=%s gold=%s tp=%s fp=%s fn=%s",
            index,
            chunk.chunk_id,
            len(predicted),
            len(gold),
            tp,
            fp,
            fn,
        )

    micro_precision = _safe_divide(totals["tp"], totals["tp"] + totals["fp"])
    micro_recall = _safe_divide(totals["tp"], totals["tp"] + totals["fn"])
    micro_f1 = _safe_divide(
        2 * micro_precision * micro_recall,
        micro_precision + micro_recall,
    )
    per_type_metrics = {
        entity_type: _metrics_for_counts(
            tp=per_type_totals[entity_type]["tp"],
            fp=per_type_totals[entity_type]["fp"],
            fn=per_type_totals[entity_type]["fn"],
        )
        for entity_type in _ENTITY_TYPES
    }

    gates = {
        "micro_precision_gte_0_85": micro_precision >= 0.85,
        "micro_recall_gte_0_30": micro_recall >= 0.30,
        "policy_precision_gte_0_80": per_type_metrics["POLICY"]["precision"] >= 0.80,
        "org_precision_gte_0_80": per_type_metrics["ORG"]["precision"] >= 0.80,
        "jurisdiction_precision_gte_0_80": per_type_metrics["JURISDICTION"]["precision"] >= 0.80,
        "person_precision_gte_0_70": per_type_metrics["PERSON"]["precision"] >= 0.70,
        "program_precision_gte_0_70": per_type_metrics["PROGRAM"]["precision"] >= 0.70,
    }

    summary: dict[str, object] = {
        "dataset_records": len(records),
        "extractor": "rule",
        "thresholds": {
            "policy": policy_threshold,
            "org": org_threshold,
            "person": person_threshold,
            "jurisdiction": jurisdiction_threshold,
            "program": program_threshold,
        },
        "total_tp": totals["tp"],
        "total_fp": totals["fp"],
        "total_fn": totals["fn"],
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "per_type": per_type_metrics,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "per_chunk": per_chunk,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        run_logger.info("evaluation report written output_path=%s", output_path)

    run_logger.info(
        (
            "evaluation completed micro_precision=%.4f micro_recall=%.4f "
            "micro_f1=%.4f all_gates_passed=%s"
        ),
        micro_precision,
        micro_recall,
        micro_f1,
        summary["all_gates_passed"],
    )
    return summary


def _load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    """Load labeled chunk dataset records from JSONL path."""
    if not dataset_path.exists():
        raise ValueError("dataset_path does not exist")

    records: list[dict[str, Any]] = []
    content = dataset_path.read_text(encoding="utf-8")
    for raw_line in content.splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        records.append(payload)
    LOGGER.debug("dataset loaded records=%s dataset_path=%s", len(records), dataset_path)
    return records


def _to_match_key(
    *,
    chunk_id: str,
    start_char: int,
    end_char: int,
    entity_type: str,
) -> tuple[str, int, int, EntityType]:
    """Build strict entity match key for exact evaluation semantics."""
    if entity_type not in _ENTITY_TYPES:
        raise ValueError("entity_type must be one of strict v1 values")
    return (chunk_id, start_char, end_char, entity_type)


def _metrics_for_counts(*, tp: int, fp: int, fn: int) -> dict[str, float | int]:
    """Compute precision, recall, and f1 for one count triple."""
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return zero when denominator is zero for metric safety."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for entity extractor evaluation script."""
    parser = argparse.ArgumentParser(
        prog="entities-eval-evaluate-extractor",
        description="Evaluate entity extractor on labeled chunks.",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to labeled entity dataset JSONL.",
    )
    parser.add_argument(
        "--policy-threshold",
        type=float,
        default=1.0,
        help="Minimum confidence threshold for POLICY mentions.",
    )
    parser.add_argument(
        "--org-threshold",
        type=float,
        default=1.0,
        help="Minimum confidence threshold for ORG mentions.",
    )
    parser.add_argument(
        "--person-threshold",
        type=float,
        default=1.0,
        help="Minimum confidence threshold for PERSON mentions.",
    )
    parser.add_argument(
        "--jurisdiction-threshold",
        type=float,
        default=1.0,
        help="Minimum confidence threshold for JURISDICTION mentions.",
    )
    parser.add_argument(
        "--program-threshold",
        type=float,
        default=1.0,
        help="Minimum confidence threshold for PROGRAM mentions.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional path to write detailed metrics JSON report.",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="warning",
        help="Log verbosity level for evaluation diagnostics.",
    )
    return parser


def main() -> int:
    """Run entity extractor evaluation from command-line arguments."""
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    output_path = Path(args.output_path) if args.output_path else None
    summary = run(
        dataset_path=Path(args.dataset_path),
        output_path=output_path,
        policy_threshold=args.policy_threshold,
        org_threshold=args.org_threshold,
        person_threshold=args.person_threshold,
        jurisdiction_threshold=args.jurisdiction_threshold,
        program_threshold=args.program_threshold,
    )
    sys.stdout.write(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
