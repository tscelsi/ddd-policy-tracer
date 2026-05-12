"""Unit tests for ML-backed claim extraction behavior."""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer

from ddd_policy_tracer.analysis.chunks.chunking_models import DocumentChunk
from ddd_policy_tracer.analysis.claims import MLClaimExtractor, MLClaimExtractorConfig


def _chunk(text: str) -> DocumentChunk:
    """Build one representative chunk fixture for extractor tests."""
    return DocumentChunk(
        chunk_id="chunk_1",
        source_id="australia_institute",
        source_document_id="https://example.org/report-1",
        document_checksum="checksum-1",
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        chunk_text=text,
    )


def _write_model_artifact(path: Path) -> None:
    """Write a deterministic token-level model artifact fixture."""
    vectorizer = DictVectorizer(sparse=True)
    sample_features = [
        {
            "token": "government",
            "token_is_title": False,
            "token_is_upper": False,
            "token_is_digit": False,
            "token_has_digit": False,
            "token_has_percent": False,
            "token_prefix_3": "gov",
            "token_suffix_3": "ent",
            "prev_token": "<start>",
            "next_token": "should",
        },
    ]
    x_matrix = vectorizer.fit_transform(sample_features)

    b_classifier = DummyClassifier(strategy="constant", constant=1)
    b_classifier.fit(x_matrix, [1])
    i_classifier = DummyClassifier(strategy="constant", constant=0)
    i_classifier.fit(x_matrix, [0])

    artifact = {
        "model_version": "claims-token-baseline-v3",
        "decision_threshold": 0.5,
        "model": {
            "vectorizer": vectorizer,
            "b_classifier": b_classifier,
            "i_classifier": i_classifier,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def test_ml_extractor_emits_span_based_claim_candidates(tmp_path: Path) -> None:
    """Predict free-form spans from ML artifact and map back to chunk text."""
    model_path = tmp_path / "claims_model.joblib"
    _write_model_artifact(model_path)
    extractor = MLClaimExtractor(config=MLClaimExtractorConfig(model_path=model_path))
    chunk = _chunk("Government should ban new coal projects.")

    claims = extractor.extract(chunk=chunk)

    assert len(claims) >= 1
    first = claims[0]
    assert first.chunk_id == "chunk_1"
    assert first.start_char >= 0
    assert first.end_char > first.start_char
    assert chunk.chunk_text[first.start_char : first.end_char] == first.evidence_text
    assert first.extractor_version == "ml-v1"


def test_ml_extractor_missing_model_path_raises_value_error(tmp_path: Path) -> None:
    """Raise clear error when configured ML artifact path is missing."""
    missing_path = tmp_path / "missing_model.joblib"
    extractor = MLClaimExtractor(config=MLClaimExtractorConfig(model_path=missing_path))
    chunk = _chunk("Government should ban new coal projects.")

    with pytest.raises(ValueError, match="claims model file not found"):
        extractor.extract(chunk=chunk)
