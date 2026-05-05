"""Unit tests for claims lightweight baseline train/eval flow."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from ddd_policy_tracer.analysis.claims.evaluation.train_baseline import run


def _write_dataset(path: Path) -> None:
    """Write deterministic claims silver dataset fixture rows."""
    rows = [
        {
            "chunk_id": "chunk_1",
            "chunk_text": "Government should ban coal projects.",
            "silver_claims": [{"start_char": 0, "end_char": 34}],
        },
        {
            "chunk_id": "chunk_2",
            "chunk_text": "Policy should reduce emissions by 4.9%.",
            "silver_claims": [{"start_char": 0, "end_char": 38}],
        },
        {
            "chunk_id": "chunk_3",
            "chunk_text": "No explicit claim here.",
            "silver_claims": [],
        },
        {
            "chunk_id": "chunk_4",
            "chunk_text": "Government should ban coal projects and reduce emissions.",
            "silver_claims": [
                {"start_char": 0, "end_char": 34},
            ],
        },
        {
            "chunk_id": "chunk_5",
            "chunk_text": "The baseline should fail this gate example.",
            "silver_claims": [{"start_char": 0, "end_char": 40}],
        },
        {
            "chunk_id": "chunk_6",
            "chunk_text": "Another row without known training claim.",
            "silver_claims": [{"start_char": 0, "end_char": 39}],
        },
        {
            "chunk_id": "chunk_7",
            "chunk_text": "Yet another plain text row.",
            "silver_claims": [],
        },
        {
            "chunk_id": "chunk_8",
            "chunk_text": "More text for splitting.",
            "silver_claims": [],
        },
        {
            "chunk_id": "chunk_9",
            "chunk_text": "Ninth row for split.",
            "silver_claims": [],
        },
        {
            "chunk_id": "chunk_10",
            "chunk_text": "Tenth row for split.",
            "silver_claims": [],
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def test_run_writes_model_artifact_and_summary(tmp_path: Path) -> None:
    """Produce model artifact, split counts, metrics, and gate outcomes."""
    dataset_path = tmp_path / "claims_silver.jsonl"
    model_output_path = tmp_path / "claims_model.joblib"
    summary_output_path = tmp_path / "claims_summary.json"
    _write_dataset(dataset_path)

    summary = run(
        dataset_path=dataset_path,
        model_output_path=model_output_path,
        summary_output_path=summary_output_path,
        seed=42,
        train_ratio=0.8,
        dev_ratio=0.1,
        test_ratio=0.1,
        precision_gate=0.5,
        recall_gate=0.1,
        f1_gate=0.1,
    )

    assert model_output_path.exists()
    artifact = joblib.load(model_output_path)
    assert artifact["model_version"] == "claims-token-baseline-v3"
    assert "model" in artifact
    assert isinstance(artifact["decision_threshold"], float)

    assert summary["split_counts"] == {"train": 8, "dev": 1, "test": 1}
    assert summary["model_version"] == "claims-token-baseline-v3"
    assert isinstance(summary["decision_threshold"], float)
    assert "metrics" in summary
    assert "gates" in summary
    assert summary_output_path.exists()
    assert isinstance(summary["metrics"]["tp"], int)
    assert isinstance(summary["metrics"]["fp"], int)
    assert isinstance(summary["metrics"]["fn"], int)


def test_run_marks_gate_failure_when_metrics_below_threshold(tmp_path: Path) -> None:
    """Set strict gates to force deterministic all_gates_passed false."""
    dataset_path = tmp_path / "claims_silver.jsonl"
    model_output_path = tmp_path / "claims_model.joblib"
    _write_dataset(dataset_path)

    summary = run(
        dataset_path=dataset_path,
        model_output_path=model_output_path,
        summary_output_path=None,
        seed=42,
        train_ratio=0.8,
        dev_ratio=0.1,
        test_ratio=0.1,
        precision_gate=1.0,
        recall_gate=1.0,
        f1_gate=1.0,
    )

    assert summary["all_gates_passed"] is False
