"""Train and evaluate a lightweight claims baseline on silver splits."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from ddd_policy_tracer.analysis.silver_dataset import deterministic_split_records


@dataclass(frozen=True)
class ClaimsBaselineModel:
    """Store memorized normalized claims used for lightweight prediction."""

    model_version: str
    memorized_claims: tuple[str, ...]

    def predict(self, *, chunk_text: str) -> set[str]:
        """Predict normalized claims present as substrings in one chunk."""
        lowered_chunk = chunk_text.casefold()
        return {
            claim
            for claim in self.memorized_claims
            if claim and claim.casefold() in lowered_chunk
        }


def run(
    *,
    dataset_path: Path,
    model_output_path: Path,
    summary_output_path: Path | None,
    seed: int,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    precision_gate: float,
    recall_gate: float,
    f1_gate: float,
) -> dict[str, object]:
    """Train lightweight claims baseline and evaluate on held-out split."""
    records = _load_records(dataset_path)
    if not records:
        raise ValueError("dataset_path contains no records")

    splits = deterministic_split_records(
        records,
        seed=seed,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
    )
    train_records = splits["train"]
    test_records = splits["test"]
    if not train_records:
        raise ValueError("train split is empty; increase dataset size or adjust ratios")

    model = _train_model(train_records)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    model_output_path.write_text(
        json.dumps(
            {
                "model_version": model.model_version,
                "memorized_claims": list(model.memorized_claims),
                "seed": seed,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics = _evaluate(model=model, records=test_records)
    gates = {
        "precision_gate": metrics["precision"] >= precision_gate,
        "recall_gate": metrics["recall"] >= recall_gate,
        "f1_gate": metrics["f1"] >= f1_gate,
    }
    summary: dict[str, object] = {
        "dataset_path": str(dataset_path),
        "model_output_path": str(model_output_path),
        "split_counts": {
            "train": len(train_records),
            "dev": len(splits["dev"]),
            "test": len(test_records),
        },
        "metrics": metrics,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }

    if summary_output_path is not None:
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_output_path.write_text(
            json.dumps(summary, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    return summary


def _train_model(records: list[dict[str, object]]) -> ClaimsBaselineModel:
    """Memorize normalized claims from the training split."""
    memorized: set[str] = set()
    for record in records:
        for claim in _record_claims(record):
            memorized.add(_normalize(claim))
    ordered = tuple(sorted(value for value in memorized if value))
    return ClaimsBaselineModel(model_version="claims-baseline-v1", memorized_claims=ordered)


def _evaluate(
    *,
    model: ClaimsBaselineModel,
    records: list[dict[str, object]],
) -> dict[str, float | int]:
    """Compute micro precision/recall/F1 over held-out records."""
    tp = 0
    fp = 0
    fn = 0
    for record in records:
        chunk_text = str(record["chunk_text"])
        predicted = model.predict(chunk_text=chunk_text)
        gold = {_normalize(claim) for claim in _record_claims(record)}
        tp += len(predicted.intersection(gold))
        fp += len(predicted - gold)
        fn += len(gold - predicted)

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


def _record_claims(record: dict[str, object]) -> list[str]:
    """Extract normalized claim text list from one silver row."""
    raw_claims = record.get("silver_claims")
    if not isinstance(raw_claims, list):
        return []
    values: list[str] = []
    for claim in raw_claims:
        if not isinstance(claim, dict):
            continue
        normalized = claim.get("normalized_claim_text")
        if isinstance(normalized, str) and normalized.strip():
            values.append(normalized)
    return values


def _load_records(dataset_path: Path) -> list[dict[str, object]]:
    """Load claims silver JSONL rows from disk."""
    if not dataset_path.exists():
        raise ValueError("dataset_path does not exist")
    records: list[dict[str, object]] = []
    for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _normalize(value: str) -> str:
    """Normalize whitespace and case for claim comparison."""
    return " ".join(value.split()).casefold()


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return zero when denominator is zero for metric safety."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for claims baseline training and evaluation."""
    parser = argparse.ArgumentParser(
        prog="claims-eval-train-baseline",
        description="Train and evaluate lightweight claims baseline model.",
    )
    parser.add_argument("--dataset-path", required=True, help="Claims silver JSONL path")
    parser.add_argument("--model-output-path", required=True, help="Model artifact JSON path")
    parser.add_argument(
        "--summary-output-path",
        default=None,
        help="Optional summary artifact JSON path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic splits")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="Dev split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--precision-gate", type=float, default=0.60, help="Precision gate")
    parser.add_argument("--recall-gate", type=float, default=0.20, help="Recall gate")
    parser.add_argument("--f1-gate", type=float, default=0.30, help="F1 gate")
    return parser


def main() -> int:
    """Run baseline training and evaluation from CLI arguments."""
    parser = _build_parser()
    args = parser.parse_args()
    summary = run(
        dataset_path=Path(args.dataset_path),
        model_output_path=Path(args.model_output_path),
        summary_output_path=Path(args.summary_output_path) if args.summary_output_path else None,
        seed=args.seed,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        precision_gate=args.precision_gate,
        recall_gate=args.recall_gate,
        f1_gate=args.f1_gate,
    )
    sys.stdout.write(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
