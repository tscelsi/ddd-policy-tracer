"""Train and evaluate a lightweight token-based claim span model."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from ddd_policy_tracer.analysis.silver_dataset import deterministic_split_records

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


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
    """Train token-based BIO model and evaluate span extraction metrics."""
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
    train_records = list(splits["train"])
    dev_records = list(splits["dev"])
    test_records = list(splits["test"])
    if not train_records:
        raise ValueError("train split is empty; increase dataset size or adjust ratios")

    model = _train_model(train_records, seed=seed)
    decision_threshold = _select_threshold(model=model, records=dev_records)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_version": "claims-token-baseline-v3",
            "seed": seed,
            "decision_threshold": decision_threshold,
            "model": model,
        },
        model_output_path,
    )

    metrics = _evaluate(model=model, records=test_records, decision_threshold=decision_threshold)
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
            "dev": len(dev_records),
            "test": len(test_records),
        },
        "model_version": "claims-token-baseline-v3",
        "decision_threshold": decision_threshold,
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


def _train_model(records: list[dict[str, object]], *, seed: int) -> dict[str, object]:
    """Train token-level classifiers for B and I tags."""
    examples = _build_token_examples(records)
    if not examples:
        raise ValueError("train split produced no token examples")

    feature_rows = [example["features"] for example in examples]
    b_labels = [example["b_label"] for example in examples]
    i_labels = [example["i_label"] for example in examples]

    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(feature_rows)

    if len(set(b_labels)) < 2:
        b_classifier: LogisticRegression | DummyClassifier = DummyClassifier(
            strategy="most_frequent",
        )
    else:
        b_classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )
    b_classifier.fit(x_train, b_labels)

    if len(set(i_labels)) < 2:
        i_classifier: LogisticRegression | DummyClassifier = DummyClassifier(
            strategy="most_frequent",
        )
    else:
        i_classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed + 1,
        )
    i_classifier.fit(x_train, i_labels)

    return {
        "vectorizer": vectorizer,
        "b_classifier": b_classifier,
        "i_classifier": i_classifier,
    }


def _select_threshold(*, model: dict[str, object], records: list[dict[str, object]]) -> float:
    """Pick a token probability threshold by dev F1."""
    if not records:
        return 0.5

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        metrics = _evaluate(model=model, records=records, decision_threshold=threshold)
        score = float(metrics["f1"])
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    return best_threshold


def _evaluate(
    *,
    model: dict[str, object],
    records: list[dict[str, object]],
    decision_threshold: float,
) -> dict[str, float | int]:
    """Compute precision/recall/F1 from exact predicted span matches."""
    tp = 0
    fp = 0
    fn = 0
    for record in records:
        chunk_id = str(record.get("chunk_id", ""))
        chunk_text = str(record.get("chunk_text", ""))
        predicted_spans = {
            (chunk_id, start_char, end_char)
            for start_char, end_char in _predict_spans(
                model=model,
                chunk_text=chunk_text,
                decision_threshold=decision_threshold,
            )
        }
        gold_spans = {
            (chunk_id, start_char, end_char)
            for start_char, end_char in _record_claim_spans(record)
        }
        tp += len(predicted_spans.intersection(gold_spans))
        fp += len(predicted_spans - gold_spans)
        fn += len(gold_spans - predicted_spans)

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


def _predict_spans(
    *,
    model: dict[str, object],
    chunk_text: str,
    decision_threshold: float,
) -> list[tuple[int, int]]:
    """Predict claim spans from token probabilities with BIO decoding."""
    tokens = _tokenize(chunk_text)
    if not tokens:
        return []

    features = [_token_features(tokens=tokens, index=index) for index in range(len(tokens))]
    vectorizer = model["vectorizer"]
    x_matrix = vectorizer.transform(features)

    b_probs = _positive_class_probabilities(model["b_classifier"], x_matrix)
    i_probs = _positive_class_probabilities(model["i_classifier"], x_matrix)

    tags: list[str] = ["O"] * len(tokens)
    for index in range(len(tokens)):
        b_score = b_probs[index]
        i_score = i_probs[index]
        if b_score >= decision_threshold:
            tags[index] = "B"
        elif i_score >= decision_threshold:
            tags[index] = "I"

    spans: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None
    for index, tag in enumerate(tags):
        token_start, token_end = tokens[index][1], tokens[index][2]
        if tag == "B":
            if current_start is not None and current_end is not None:
                spans.append((current_start, current_end))
            current_start = token_start
            current_end = token_end
            continue

        if tag == "I" and current_start is not None:
            current_end = token_end
            continue

        if current_start is not None and current_end is not None:
            spans.append((current_start, current_end))
        current_start = None
        current_end = None

    if current_start is not None and current_end is not None:
        spans.append((current_start, current_end))
    return _dedupe_spans(spans)


def _dedupe_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Dedupe predicted spans while preserving insertion order."""
    seen: set[tuple[int, int]] = set()
    deduped: list[tuple[int, int]] = []
    for span in spans:
        if span in seen:
            continue
        seen.add(span)
        deduped.append(span)
    return deduped


def _build_token_examples(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build token-level BIO examples from silver claim spans."""
    examples: list[dict[str, object]] = []
    for record in records:
        chunk_text = str(record.get("chunk_text", ""))
        tokens = _tokenize(chunk_text)
        if not tokens:
            continue
        gold_tags = _gold_bio_tags(tokens=tokens, claim_spans=_record_claim_spans(record))
        for index, (token_text, _, _) in enumerate(tokens):
            features = _token_features(tokens=tokens, index=index)
            examples.append(
                {
                    "token_text": token_text,
                    "features": features,
                    "b_label": 1 if gold_tags[index] == "B" else 0,
                    "i_label": 1 if gold_tags[index] == "I" else 0,
                },
            )
    return examples


def _gold_bio_tags(
    *,
    tokens: list[tuple[str, int, int]],
    claim_spans: list[tuple[int, int]],
) -> list[str]:
    """Assign BIO tags to tokens given gold claim spans."""
    tags = ["O"] * len(tokens)
    for claim_start, claim_end in claim_spans:
        overlapping_indexes = [
            index
            for index, (_, token_start, token_end) in enumerate(tokens)
            if token_start < claim_end and token_end > claim_start
        ]
        if not overlapping_indexes:
            continue
        first_index = overlapping_indexes[0]
        if tags[first_index] == "O":
            tags[first_index] = "B"
        for index in overlapping_indexes[1:]:
            if tags[index] == "O":
                tags[index] = "I"
    return tags


def _tokenize(text: str) -> list[tuple[str, int, int]]:
    """Tokenize chunk text into token strings and char offsets."""
    tokens: list[tuple[str, int, int]] = []
    for match in _TOKEN_RE.finditer(text):
        tokens.append((match.group(0), match.start(), match.end()))
    return tokens


def _token_features(
    *,
    tokens: list[tuple[str, int, int]],
    index: int,
) -> dict[str, object]:
    """Build lightweight contextual token features for one index."""
    token_text = tokens[index][0]
    prev_text = tokens[index - 1][0] if index > 0 else "<START>"
    next_text = tokens[index + 1][0] if index < len(tokens) - 1 else "<END>"

    return {
        "token": token_text.casefold(),
        "token_is_title": token_text.istitle(),
        "token_is_upper": token_text.isupper(),
        "token_is_digit": token_text.isdigit(),
        "token_has_digit": any(char.isdigit() for char in token_text),
        "token_has_percent": "%" in token_text,
        "token_prefix_3": token_text[:3].casefold(),
        "token_suffix_3": token_text[-3:].casefold(),
        "prev_token": prev_text.casefold(),
        "next_token": next_text.casefold(),
    }


def _positive_class_probabilities(classifier: object, x_matrix: object) -> list[float]:
    """Return class-1 probabilities for a fitted sklearn classifier."""
    probabilities = classifier.predict_proba(x_matrix)
    classes = getattr(classifier, "classes_", None)
    if classes is None:
        return [0.0 for _ in range(len(probabilities))]
    classes_list = list(classes)
    if 1 not in classes_list:
        return [0.0 for _ in range(len(probabilities))]
    class_index = classes_list.index(1)
    return [float(row[class_index]) for row in probabilities]


def _record_claim_spans(record: dict[str, object]) -> list[tuple[int, int]]:
    """Extract valid claim spans from one span-based silver row."""
    raw_claims = record.get("silver_claims")
    if not isinstance(raw_claims, list):
        return []
    chunk_text = str(record.get("chunk_text", ""))
    values: list[tuple[int, int]] = []
    for claim in raw_claims:
        if not isinstance(claim, dict):
            continue
        start_char = claim.get("start_char")
        end_char = claim.get("end_char")
        if not isinstance(start_char, int) or not isinstance(end_char, int):
            continue
        if start_char < 0 or end_char > len(chunk_text) or end_char <= start_char:
            continue
        values.append((start_char, end_char))
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


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return zero when denominator is zero for metric safety."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for token-level claims baseline training and evaluation."""
    parser = argparse.ArgumentParser(
        prog="claims-eval-train-baseline",
        description="Train and evaluate lightweight token-level claims baseline model.",
    )
    parser.add_argument("--dataset-path", required=True, help="Claims silver JSONL path")
    parser.add_argument("--model-output-path", required=True, help="Model artifact joblib path")
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
    """Run token-level baseline training and evaluation from CLI arguments."""
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
