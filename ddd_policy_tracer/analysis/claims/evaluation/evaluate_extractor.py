"""Evaluate one claim extractor against a labeled chunk dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal

from ddd_policy_tracer.analysis.chunking_models import DocumentChunk
from ddd_policy_tracer.analysis.claims.extractors import (
    LLMClaimExtractor,
    LLMClaimExtractorConfig,
    RuleBasedClaimExtractorConfig,
    RuleBasedSentenceClaimExtractor,
)
from ddd_policy_tracer.analysis.claims.ports import ClaimExtractor
from ddd_policy_tracer.utils.logger import configure_logging, get_logger

LOGGER = get_logger(__name__, ctx="claims_evaluation")


def run(
    *,
    dataset_path: Path,
    extractor_kind: Literal["rule", "llm"],
    threshold: float,
    output_path: Path | None,
    extractor: ClaimExtractor,
) -> dict[str, object]:
    """Evaluate extractor precision/recall metrics over labeled chunks."""
    records = _load_dataset(dataset_path)
    run_logger = LOGGER.bind(
        extractor=extractor_kind,
        threshold=threshold,
        dataset_path=str(dataset_path),
        dataset_records=len(records),
    )
    run_logger.info("evaluation started")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_chunk: list[dict[str, object]] = []

    for record in records:
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
            _normalize_text(claim.normalized_claim_text)
            for claim in extractor.extract(chunk=chunk)
        }
        gold = {
            _normalize_text(claim["normalized_claim_text"])
            for claim in record["gold_claims"]
        }

        tp = len(predicted.intersection(gold))
        fp = len(predicted - gold)
        fn = len(gold - predicted)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_chunk.append(
            {
                "chunk_id": chunk.chunk_id,
                "predicted_claims": len(predicted),
                "gold_claims": len(gold),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
        )
        run_logger.debug(
            "chunk evaluated chunk_id=%s predicted=%s gold=%s tp=%s fp=%s fn=%s",
            chunk.chunk_id,
            len(predicted),
            len(gold),
            tp,
            fp,
            fn,
        )

    micro_precision = _safe_divide(total_tp, total_tp + total_fp)
    micro_recall = _safe_divide(total_tp, total_tp + total_fn)
    micro_f1 = _safe_divide(
        2 * micro_precision * micro_recall,
        micro_precision + micro_recall,
    )
    summary: dict[str, object] = {
        "dataset_records": len(records),
        "extractor": extractor_kind,
        "threshold": threshold,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
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
        "evaluation completed micro_precision=%.4f micro_recall=%.4f micro_f1=%.4f",
        micro_precision,
        micro_recall,
        micro_f1,
    )

    return summary


def _load_dataset(dataset_path: Path) -> list[dict[str, object]]:
    """Load labeled chunk dataset records from JSONL path."""
    if not dataset_path.exists():
        raise ValueError("dataset_path does not exist")

    records: list[dict[str, object]] = []
    content = dataset_path.read_text(encoding="utf-8")
    for raw_line in content.splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        records.append(payload)
    LOGGER.debug("dataset loaded records=%s dataset_path=%s", len(records), dataset_path)
    return records


def _normalize_text(value: str) -> str:
    """Normalize claim text for deterministic comparison semantics."""
    return " ".join(value.split()).casefold()


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return zero when denominator is zero for metric safety."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for claim extractor evaluation script."""
    parser = argparse.ArgumentParser(
        prog="claims-eval-evaluate-extractor",
        description="Evaluate claim extractor on labeled chunks.",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to labeled chunk dataset JSONL.",
    )
    parser.add_argument(
        "--extractor",
        choices=["rule", "llm"],
        default=None,
        help="Extractor strategy override. Falls back to CLAIMS_EXTRACTOR env var.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Rule-based extractor threshold for evaluation run.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4.1-mini",
        help="Model used by LLM extractor.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="Temperature used by LLM extractor.",
    )
    parser.add_argument(
        "--llm-max-claims-per-chunk",
        type=int,
        default=8,
        help="Maximum claims requested from LLM per chunk.",
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


def _build_extractor(
    *,
    extractor_kind: Literal["rule", "llm"],
    threshold: float,
    llm_model: str,
    llm_temperature: float,
    llm_max_claims_per_chunk: int,
) -> ClaimExtractor:
    """Build one extractor strategy for evaluation execution."""
    if extractor_kind == "llm":
        return LLMClaimExtractor(
            config=LLMClaimExtractorConfig(
                model=llm_model,
                temperature=llm_temperature,
                max_claims_per_chunk=llm_max_claims_per_chunk,
            ),
        )

    return RuleBasedSentenceClaimExtractor(
        RuleBasedClaimExtractorConfig(threshold=threshold),
    )


def main() -> int:
    """Run extractor evaluation from command-line arguments."""
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    env_extractor = os.environ.get("CLAIMS_EXTRACTOR", "rule")
    extractor_kind = args.extractor or env_extractor
    if extractor_kind not in {"rule", "llm"}:
        raise ValueError("CLAIMS_EXTRACTOR must be either 'rule' or 'llm'")

    output_path = Path(args.output_path) if args.output_path else None
    extractor = _build_extractor(
        extractor_kind=extractor_kind,
        threshold=args.threshold,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_max_claims_per_chunk=args.llm_max_claims_per_chunk,
    )
    summary = run(
        dataset_path=Path(args.dataset_path),
        extractor_kind=extractor_kind,
        threshold=args.threshold,
        output_path=output_path,
        extractor=extractor,
    )
    sys.stdout.write(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
