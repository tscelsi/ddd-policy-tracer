"""Unit tests for Hugging Face claim extractor evaluation workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ddd_policy_tracer.analysis.chunks.chunking_models import DocumentChunk
from ddd_policy_tracer.analysis.claims.ml.evaluate_hf_extractor import run


def _write_dataset(path: Path) -> None:
    """Write one deterministic span-based claims silver dataset fixture."""
    row = {
        "chunk_id": "chunk_1",
        "source_id": "australia_institute",
        "source_document_id": "https://example.org/report-1",
        "document_checksum": "checksum-1",
        "chunk_text": "Government should ban new coal projects.",
        "silver_claims": [{"start_char": 0, "end_char": 39}],
    }
    path.write_text(json.dumps(row, ensure_ascii=True) + "\n", encoding="utf-8")


def test_run_scores_exact_span_metrics_with_stubbed_hf_extractor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Evaluate rows and emit deterministic precision/recall/F1 summary."""
    dataset_path = tmp_path / "claims_dataset.jsonl"
    summary_path = tmp_path / "hf_eval_summary.json"
    _write_dataset(dataset_path)

    class StubHFExtractor:
        """Emit one deterministic predicted span for each chunk."""

        def __init__(self, config: object) -> None:
            """Store config to match constructor contract for patching."""
            self._config = config

        def extract(self, *, chunk: DocumentChunk) -> list[object]:
            """Return one matched claim candidate-style object."""

            class _Claim:
                """Minimal span carrier used by evaluator."""

                start_char = 0
                end_char = 39

            return [_Claim()]

    monkeypatch.setattr(
        "ddd_policy_tracer.analysis.claims.ml.evaluate_hf_extractor.HuggingFaceClaimExtractor",
        StubHFExtractor,
    )

    summary = run(
        dataset_path=dataset_path,
        model_name="Babelscape/t5-base-summarization-claim-extractor",
        max_rows=None,
        summary_output_path=summary_path,
    )

    assert summary["rows_loaded"] == 1
    assert summary["rows_evaluated"] == 1
    assert summary["metrics"]["tp"] == 1
    assert summary["metrics"]["fp"] == 0
    assert summary["metrics"]["fn"] == 0
    assert summary["metrics"]["precision"] == 1.0
    assert summary["metrics"]["recall"] == 1.0
    assert summary["metrics"]["f1"] == 1.0
    assert summary_path.exists()
