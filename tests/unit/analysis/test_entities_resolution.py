"""Unit tests for deterministic entity judge and status policy."""

from __future__ import annotations

from ddd_policy_tracer.analysis.entities import EntityMention
from ddd_policy_tracer.analysis.entities.resolution import (
    DeterministicEntityJudge,
    apply_resolution_decision,
)
from ddd_policy_tracer.analysis.entities.retrieval import RetrievalCandidate, RetrievalResult


def _mention(*, mention_text: str = "Clean Energy Act") -> EntityMention:
    """Build one mention fixture for resolution testing."""
    return EntityMention(
        entity_id="entity_1",
        chunk_id="chunk_1",
        source_id="australia_institute",
        source_document_id="https://example.org/report-1",
        document_checksum="checksum-1",
        start_char=0,
        end_char=len(mention_text),
        mention_text=mention_text,
        normalized_mention_text=mention_text,
        entity_type="POLICY",
        confidence=0.9,
        extractor_version="robust-ensemble-v1",
    )


def _candidate(*, score: float, key: str, entity_type: str = "POLICY") -> RetrievalCandidate:
    """Build one retrieval candidate fixture with score fields."""
    return RetrievalCandidate(
        canonical_entity_key=key,
        canonical_name="Clean Energy Act",
        entity_type=entity_type,
        fused_score=score,
        lexical_score=score,
        vector_score=score,
        diagnostics={},
    )


def test_judge_returns_linked_for_high_confidence_match() -> None:
    """Mark decision as linked when calibrated score exceeds linked threshold."""
    judge = DeterministicEntityJudge()
    mention = _mention()
    result = RetrievalResult(
        mention_text=mention.mention_text,
        normalized_mention_text=mention.normalized_mention_text,
        mention_entity_type=mention.entity_type,
        candidates=[_candidate(score=0.9, key="policy:clean-energy-act")],
    )

    decision = judge.evaluate(mention=mention, retrieval_result=result)

    assert decision.decision_status == "linked"
    assert decision.decision_score >= 0.85
    assert decision.selected_candidate_key == "policy:clean-energy-act"


def test_judge_returns_needs_review_for_midrange_score() -> None:
    """Mark decision as needs_review when score falls in review band."""
    judge = DeterministicEntityJudge()
    mention = _mention()
    result = RetrievalResult(
        mention_text=mention.mention_text,
        normalized_mention_text=mention.normalized_mention_text,
        mention_entity_type=mention.entity_type,
        candidates=[
            _candidate(score=0.62, key="policy:clean-energy-act"),
            _candidate(score=0.58, key="program:clean-energy-program", entity_type="PROGRAM"),
        ],
    )

    decision = judge.evaluate(mention=mention, retrieval_result=result)

    assert decision.decision_status == "needs_review"
    assert 0.60 <= decision.decision_score < 0.85


def test_judge_returns_new_candidate_for_low_score_valid_mention() -> None:
    """Mark low-score but valid mention as new candidate."""
    judge = DeterministicEntityJudge()
    mention = _mention(mention_text="Future Adaptation Plan")
    result = RetrievalResult(
        mention_text=mention.mention_text,
        normalized_mention_text=mention.normalized_mention_text,
        mention_entity_type=mention.entity_type,
        candidates=[_candidate(score=0.2, key="policy:clean-energy-act")],
    )

    decision = judge.evaluate(mention=mention, retrieval_result=result)

    assert decision.decision_status == "new_candidate"
    assert decision.decision_score < 0.60


def test_judge_returns_abstain_when_no_candidates() -> None:
    """Abstain when retrieval does not provide any candidate evidence."""
    judge = DeterministicEntityJudge()
    mention = _mention(mention_text="X1")
    result = RetrievalResult(
        mention_text=mention.mention_text,
        normalized_mention_text=mention.normalized_mention_text,
        mention_entity_type=mention.entity_type,
        candidates=[],
    )

    decision = judge.evaluate(mention=mention, retrieval_result=result)

    assert decision.decision_status == "abstain"
    assert decision.decision_score == 0.0


def test_apply_resolution_decision_sets_metadata_and_canonical_key() -> None:
    """Persist decision metadata and linked key on mention output."""
    judge = DeterministicEntityJudge()
    mention = _mention()
    result = RetrievalResult(
        mention_text=mention.mention_text,
        normalized_mention_text=mention.normalized_mention_text,
        mention_entity_type=mention.entity_type,
        candidates=[_candidate(score=0.9, key="policy:clean-energy-act")],
    )

    decision = judge.evaluate(mention=mention, retrieval_result=result)
    resolved = apply_resolution_decision(mention=mention, decision=decision)

    assert resolved.canonical_entity_key == "policy:clean-energy-act"
    assert resolved.metadata is not None
    assert resolved.metadata["decision_status"] == "linked"
    assert resolved.metadata["selected_candidate_key"] == "policy:clean-energy-act"
