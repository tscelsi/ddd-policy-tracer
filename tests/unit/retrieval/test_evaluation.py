"""Tests for reviewed evidence cases and retrieval evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from ddd_policy_tracer.retrieval.evaluation import (
    evaluate_cases,
    transition_review_status,
    validate_case,
)


class FakeEmbedder:
    """Provide deterministic query vectors for retrieval evaluation."""

    def embed_one(
        self,
        text: str,
        prompt_name: str = "document",
        return_tensor: bool = False,
    ) -> list[float]:
        """Return a stable vector for the requested query."""
        return [1.0]


class FakeChunk:
    """Represent one result returned by the fake retriever."""

    def __init__(self, chunk_id: str) -> None:
        """Keep the chunk identity used to report target ranks."""
        self.chunk_id = chunk_id
        self.chunk_text = "evidence"
        self.source_document_id = "speech"
        self.metadata = None


class FakeRetriever:
    """Return a controlled ranking for each query."""

    def __init__(
        self,
        ranked_ids: Sequence[str],
        corpus_ids: set[str] | None = None,
    ) -> None:
        """Store the ranked chunk identities."""
        self.ranked_ids = ranked_ids
        self.corpus_ids = set(ranked_ids) if corpus_ids is None else corpus_ids

    def search_hybrid(
        self,
        *,
        query: str,
        query_embedding: Sequence[float],
        limit: int = 10,
        candidate_limit: int | None = None,
        filters: object | None = None,
    ) -> list[FakeChunk]:
        """Return the configured top-k chunk results."""
        return [FakeChunk(chunk_id) for chunk_id in self.ranked_ids[:limit]]

    def existing_chunk_ids(self, chunk_ids: Sequence[str]) -> set[str]:
        """Return the requested IDs that remain in the fake corpus."""
        return set(chunk_ids) & self.corpus_ids


def answerable_case(**overrides: object) -> dict[str, object]:
    """Build a reviewed answerable retrieval case."""
    case: dict[str, object] = {
        "case_id": "case-one",
        "question": "What does the speaker say?",
        "answerable": True,
        "gold_evidence_chunk_ids": ["chunk-a", "chunk-b"],
        "review_status": "approved",
        "review_provenance": {"reviewer": "analyst", "status": "approved"},
    }
    return case | overrides


def test_case_schema_rejects_answerable_case_without_required_evidence() -> None:
    """Require one or more conjunctive evidence targets for answerable cases."""
    with pytest.raises(ValueError, match="gold_evidence_chunk_ids"):
        validate_case(answerable_case(gold_evidence_chunk_ids=[]))


def test_case_schema_requires_review_provenance_for_approval() -> None:
    """Prevent status edits from bypassing the human review record."""
    with pytest.raises(ValueError, match="review_provenance"):
        validate_case(answerable_case(review_provenance=None))


def test_case_schema_requires_confirmed_absence_reason_for_unanswerable_cases() -> None:
    """Keep unanswerable cases tied to a human-documented corpus absence."""
    with pytest.raises(ValueError, match="abstention_reason"):
        validate_case(
            {
                "case_id": "not-in-corpus",
                "question": "What happened?",
                "answerable": False,
                "gold_evidence_chunk_ids": [],
                "review_status": "approved",
                "review_provenance": {"reviewer": "analyst", "status": "approved"},
            },
        )


def test_review_transition_records_human_decision_without_auto_approval() -> None:
    """Keep generated questions as drafts until a reviewer explicitly approves them."""
    draft = answerable_case(review_status="draft")

    reviewed = transition_review_status(
        draft,
        status="approved",
        reviewer="analyst",
        note="All required passages directly support this query.",
    )

    assert reviewed["review_status"] == "approved"
    provenance = reviewed["review_provenance"]
    assert provenance["reviewer"] == "analyst"
    assert provenance["status"] == "approved"
    assert provenance["note"] == "All required passages directly support this query."
    assert provenance["reviewed_at"]


def test_rejected_case_requires_reason() -> None:
    """Record why a candidate query was rejected."""
    with pytest.raises(ValueError, match="rejection"):
        transition_review_status(
            answerable_case(review_status="draft"),
            status="rejected",
            reviewer="analyst",
        )


def test_evaluation_requires_approved_answerable_cases() -> None:
    """Never report an empty evaluation as a successful run."""
    with pytest.raises(ValueError, match="approved answerable"):
        evaluate_cases(
            [answerable_case(review_status="draft")],
            retriever=FakeRetriever([]),
            chunk_index=FakeRetriever([]),
            embedding_service=FakeEmbedder(),
        )


def test_evaluation_requires_all_gold_evidence_and_reports_ranks() -> None:
    """Pass only when every required chunk is in the top ten; extras are allowed."""
    result = evaluate_cases(
        [answerable_case()],
        retriever=FakeRetriever(["extra", "chunk-b", "chunk-a"]),
        chunk_index=FakeRetriever(["extra", "chunk-b", "chunk-a"]),
        embedding_service=FakeEmbedder(),
    )

    assert result.answerable_passed == 1
    assert result.answerable_total == 1
    assert result.cases[0].target_ranks == {"chunk-a": 3, "chunk-b": 2}


def test_evaluation_fails_for_stale_gold_ids_without_remapping() -> None:
    """Report annotated IDs absent from the corpus as stale and fail the case."""
    result = evaluate_cases(
        [answerable_case()],
        retriever=FakeRetriever(["replacement"]),
        chunk_index=FakeRetriever(["replacement"], corpus_ids={"chunk-a"}),
        embedding_service=FakeEmbedder(),
    )

    assert result.answerable_failed == 1
    assert result.cases[0].stale_chunk_ids == ("chunk-b",)
    assert result.cases[0].target_ranks == {"chunk-a": None, "chunk-b": None}


def test_approved_unanswerable_cases_are_reported_separately() -> None:
    """Report human-confirmed corpus-absence cases without applying a threshold."""
    case = {
        "case_id": "not-in-corpus",
        "question": "What happened?",
        "answerable": False,
        "gold_evidence_chunk_ids": [],
        "abstention_reason": "The indexed corpus does not include that date.",
        "absence_confirmed_by": "analyst",
        "review_status": "approved",
        "review_provenance": {"reviewer": "analyst", "status": "approved"},
    }

    result = evaluate_cases(
        [answerable_case(), case],
        retriever=FakeRetriever(["chunk-a", "chunk-b"]),
        chunk_index=FakeRetriever(["chunk-a", "chunk-b"]),
        embedding_service=FakeEmbedder(),
    )

    assert result.unanswerable_case_ids == ("not-in-corpus",)
