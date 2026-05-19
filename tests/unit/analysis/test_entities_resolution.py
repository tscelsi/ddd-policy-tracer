"""Unit tests for deterministic entity judge and status policy."""

from __future__ import annotations

from ddd_policy_tracer.analysis.entities import EntityMention
from ddd_policy_tracer.analysis.entities.resolution import (
    CatalogEligibilityPolicy,
    DeterministicEntityJudge,
    apply_resolution_decision,
)
from ddd_policy_tracer.analysis.entities.retrieval import RetrievalCandidate, RetrievalResult


def _mention(
    *,
    mention_text: str = "Australia",
    entity_type: str = "JURISDICTION",
) -> EntityMention:
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
        entity_type=entity_type,
        confidence=0.9,
        extractor_version="robust-ensemble-v1",
    )


def _candidate(*, score: float, key: str, entity_type: str = "JURISDICTION") -> RetrievalCandidate:
    """Build one retrieval candidate fixture with score fields."""
    return RetrievalCandidate(
        canonical_entity_key=key,
        canonical_name="Australia",
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
            candidates=[_candidate(score=0.9, key="jurisdiction:australia")],
        )

    decision = judge.evaluate(mention=mention, retrieval_result=result)

    assert decision.decision_status == "linked"
    assert decision.decision_score >= 0.85
    assert decision.selected_candidate_key == "jurisdiction:australia"


def test_judge_returns_needs_review_for_midrange_score() -> None:
    """Mark decision as needs_review when score falls in review band."""
    judge = DeterministicEntityJudge()
    mention = _mention()
    result = RetrievalResult(
            mention_text=mention.mention_text,
            normalized_mention_text=mention.normalized_mention_text,
            mention_entity_type=mention.entity_type,
            candidates=[
                _candidate(score=0.62, key="jurisdiction:australia"),
                _candidate(score=0.58, key="org:australian-government", entity_type="ORG"),
            ],
        )

    decision = judge.evaluate(mention=mention, retrieval_result=result)

    assert decision.decision_status == "needs_review"
    assert 0.60 <= decision.decision_score < 0.90
    assert decision.selected_candidate_key is None


def test_judge_returns_new_candidate_for_low_score_valid_mention() -> None:
    """Mark low-score but valid mention as new candidate."""
    judge = DeterministicEntityJudge()
    mention = _mention(mention_text="Alice Brown", entity_type="PERSON")
    result = RetrievalResult(
        mention_text=mention.mention_text,
        normalized_mention_text=mention.normalized_mention_text,
        mention_entity_type=mention.entity_type,
        candidates=[_candidate(score=0.2, key="person:other-name", entity_type="PERSON")],
    )

    decision = judge.evaluate(mention=mention, retrieval_result=result)

    assert decision.decision_status == "new_candidate"
    assert decision.decision_score < 0.60


def test_judge_returns_abstain_when_no_candidates() -> None:
    """Abstain when retrieval does not provide any candidate evidence."""
    judge = DeterministicEntityJudge()
    mention = _mention(mention_text="X1", entity_type="PERSON")
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
            candidates=[_candidate(score=0.9, key="jurisdiction:australia")],
        )

    decision = judge.evaluate(mention=mention, retrieval_result=result)
    resolved = apply_resolution_decision(mention=mention, decision=decision)

    assert resolved.canonical_entity_key == "jurisdiction:australia"
    assert resolved.metadata is not None
    assert resolved.metadata["decision_status"] == "linked"
    assert resolved.metadata["selected_candidate_key"] == "jurisdiction:australia"
    assert resolved.metadata["max_candidate_score"] == 0.9
    candidate_links = resolved.metadata["candidate_links"]
    assert isinstance(candidate_links, list)
    assert candidate_links[0]["canonical_entity_key"] == "jurisdiction:australia"
    assert candidate_links[0]["confidence_score"] == 0.9


def test_eligibility_policy_marks_policy_mentions_ineligible_for_catalog() -> None:
    """Reject policy entities from catalog curation eligibility."""
    policy = CatalogEligibilityPolicy()

    decision = policy.evaluate(mention=_mention(mention_text="Clean Energy Act", entity_type="POLICY"))

    assert decision.is_eligible is False
    assert "ineligible_entity_type" in decision.reason_codes


def test_eligibility_policy_rejects_ambiguous_generic_terms() -> None:
    """Reject generic terms like Government as non-unique entities."""
    policy = CatalogEligibilityPolicy()

    decision = policy.evaluate(mention=_mention(mention_text="Government", entity_type="ORG"))

    assert decision.is_eligible is False
    assert "ambiguous_generic" in decision.reason_codes
