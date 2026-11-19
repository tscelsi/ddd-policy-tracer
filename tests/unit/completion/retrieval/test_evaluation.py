"""Tests for reviewed evidence cases and retrieval evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pytest

from ddd_policy_tracer.completion.filtering.metadata_filters import (
    MetadataFilter,
    MetadataFilterExtraction,
)
from ddd_policy_tracer.completion.filtering.resolvers import SpeakerResolution
from ddd_policy_tracer.completion._evaluation.main import (
    evaluate_cases,
    transition_review_status,
    validate_case,
)
from ddd_policy_tracer.completion.service_layer import (
    RagCompletionService,
    UnsupportedFilterError,
)
from ddd_policy_tracer.completion.synthesis.answer_synthesizer import AnswerSynthesizer
from ddd_policy_tracer.completion.synthesis.models import AnswerDraft, EvidenceChunk


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
        filters: MetadataFilter | None = None,
    ) -> list[FakeChunk]:
        """Return the configured top-k chunk results."""
        return [FakeChunk(chunk_id) for chunk_id in self.ranked_ids[:limit]]

    def existing_chunk_ids(self, chunk_ids: Sequence[str]) -> set[str]:
        """Return the requested IDs that remain in the fake corpus."""
        return set(chunk_ids) & self.corpus_ids


class FakeMetadataFilterExtractor:
    """Return no filters while tracking metadata extraction calls."""

    def __init__(self, fail_on_question: str | None = None) -> None:
        """Start the extraction count at zero."""
        self.call_count = 0
        self.fail_on_question = fail_on_question

    def extract(
        self,
        question: str,
        *,
        reference_date: date,
    ) -> MetadataFilterExtraction:
        """Return an empty filter result for deterministic evaluation tests."""
        self.call_count += 1
        if question == self.fail_on_question:
            raise UnsupportedFilterError("Unsupported date constraint")
        return MetadataFilterExtraction()


class FakeSpeakerResolver:
    """Resolve no requested speaker names for evaluation tests."""

    async def resolve(self, speaker_names: Sequence[str]) -> SpeakerResolution:
        """Return an empty resolution when no speaker filter is requested."""
        return SpeakerResolution()


class FakeAnswerGenerator:
    """Provide an unused answer generator required by the completion service."""

    def generate(self, question: str, evidence: Sequence[EvidenceChunk]) -> AnswerDraft:
        """Return an empty draft; evaluation only exercises retrieval."""
        return AnswerDraft()


def build_completion_service(
    retriever: FakeRetriever,
    metadata_filter_extractor: FakeMetadataFilterExtractor | None = None,
) -> tuple[RagCompletionService, FakeMetadataFilterExtractor]:
    """Wire fake retrieval and metadata ports through the production completion service."""
    extractor = metadata_filter_extractor or FakeMetadataFilterExtractor()
    return (
        RagCompletionService(
            retriever=retriever,
            embedding_service=FakeEmbedder(),
            synthesizer=AnswerSynthesizer(FakeAnswerGenerator()),
            metadata_filter_extractor=extractor,
            speaker_name_resolver=FakeSpeakerResolver(),
        ),
        extractor,
    )


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


async def test_evaluation_requires_approved_answerable_cases() -> None:
    """Never report an empty evaluation as a successful run."""
    retriever = FakeRetriever([])
    completion_service, _ = build_completion_service(retriever)
    with pytest.raises(ValueError, match="approved answerable"):
        await evaluate_cases(
            [answerable_case(review_status="draft")],
            completion_service=completion_service,
            chunk_index=retriever,
        )


async def test_evaluation_uses_completion_retrieval_and_reports_ranks() -> None:
    """Evaluate the same metadata-filtered ranked chunks used by RAG completion."""
    retriever = FakeRetriever(["extra", "chunk-b", "chunk-a"])
    completion_service, extractor = build_completion_service(retriever)
    result = await evaluate_cases(
        [answerable_case()],
        completion_service=completion_service,
        chunk_index=retriever,
    )

    assert extractor.call_count == 1
    assert result.answerable_passed == 1
    assert result.answerable_total == 1
    assert result.cases[0].target_ranks == {"chunk-a": 3, "chunk-b": 2}
    assert [
        (
            chunk.rank,
            chunk.chunk_id,
            chunk.chunk_text,
            chunk.source_document_id,
            chunk.metadata,
        )
        for chunk in result.cases[0].retrieved_chunks
    ] == [
        (1, "extra", "evidence", "speech", None),
        (2, "chunk-b", "evidence", "speech", None),
        (3, "chunk-a", "evidence", "speech", None),
    ]


async def test_evaluation_records_retrieval_errors_and_continues_to_next_case() -> None:
    """Mark a query failed when retrieval raises and continue evaluating later queries."""
    retriever = FakeRetriever(["chunk-a", "chunk-b"])
    extractor = FakeMetadataFilterExtractor(fail_on_question="Unsupported filter request")
    completion_service, _ = build_completion_service(retriever, extractor)
    cases = [
        answerable_case(
            case_id="unsupported-case",
            question="Unsupported filter request",
            gold_evidence_chunk_ids=["chunk-a"],
        ),
        answerable_case(
            case_id="following-case",
            question="What does the speaker say?",
            gold_evidence_chunk_ids=["chunk-b"],
        ),
    ]

    report = await evaluate_cases(
        cases,
        completion_service=completion_service,
        chunk_index=retriever,
    )

    assert report.answerable_total == 2
    assert report.answerable_failed == 1
    assert report.cases[0].error == "UnsupportedFilterError: Unsupported date constraint"
    assert report.cases[0].target_ranks == {"chunk-a": None}
    assert report.cases[1].passed
    assert report.cases[1].target_ranks == {"chunk-b": 2}


async def test_evaluation_fails_for_stale_gold_ids_without_remapping() -> None:
    """Report annotated IDs absent from the corpus as stale and fail the case."""
    retriever = FakeRetriever(["replacement"], corpus_ids={"chunk-a"})
    completion_service, _ = build_completion_service(retriever)
    result = await evaluate_cases(
        [answerable_case()],
        completion_service=completion_service,
        chunk_index=retriever,
    )

    assert result.answerable_failed == 1
    assert result.cases[0].stale_chunk_ids == ("chunk-b",)
    assert result.cases[0].target_ranks == {"chunk-a": None, "chunk-b": None}


async def test_approved_unanswerable_cases_are_reported_separately() -> None:
    """Report human-confirmed corpus-absence cases without applying a threshold."""
    case: dict[str, object] = {
        "case_id": "not-in-corpus",
        "question": "What happened?",
        "answerable": False,
        "gold_evidence_chunk_ids": [],
        "abstention_reason": "The indexed corpus does not include that date.",
        "absence_confirmed_by": "analyst",
        "review_status": "approved",
        "review_provenance": {"reviewer": "analyst", "status": "approved"},
    }

    retriever = FakeRetriever(["chunk-a", "chunk-b"])
    completion_service, _ = build_completion_service(retriever)
    result = await evaluate_cases(
        [answerable_case(), case],
        completion_service=completion_service,
        chunk_index=retriever,
    )

    assert result.unanswerable_case_ids == ("not-in-corpus",)
