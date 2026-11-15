"""Tests for synthesizing answers grounded in retrieved speech evidence."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from ddd_policy_tracer.answering.service_layer import (
    AnswerClaim,
    AnswerDraft,
    AnswerSynthesizer,
    EvidenceChunk,
    InvalidEvidenceReferenceError,
)
from ddd_policy_tracer.retrieval.sqlite_retrieval import SQLiteChunkResult


class StubAnswerGenerator:
    """Return a prepared answer draft for synthesis tests."""

    def __init__(self, draft: AnswerDraft) -> None:
        """Store the answer draft supplied by the test."""
        self.draft = draft
        self.called = False

    def generate(self, question: str, evidence: Sequence[EvidenceChunk]) -> AnswerDraft:
        """Return the prepared answer draft without calling a model."""
        self.called = True
        return self.draft


def _speech_chunk(
    *,
    chunk_id: str = "chunk-haines",
    source_document_id: str = "282335_2025-02-12_hor_main_16_0_0_7_0",
    chunk_text: str = "I support the bill because it helps children and parents.",
) -> SQLiteChunkResult:
    """Build a retrieved speech chunk with citation provenance."""
    return SQLiteChunkResult(
        chunk_id=chunk_id,
        source_id="hansard",
        source_document_id=source_document_id,
        chunk_index=0,
        start_char=0,
        end_char=len(chunk_text),
        chunk_text=chunk_text,
        topics=None,
        metadata='{"doc_metadata":{"talker":{"name":"Helen Haines"}}}',
    )


def test_synthesis_renders_a_cited_claim_with_source_provenance() -> None:
    """Render a generated claim with a verifiable inline chunk citation."""
    evidence = _speech_chunk()
    generator = StubAnswerGenerator(
        AnswerDraft(
            claims=(
                AnswerClaim(
                    text=(
                        "Helen Haines supports the bill because she says it helps children "
                        "and parents."
                    ),
                    evidence_chunk_ids=(evidence.chunk_id,),
                ),
            ),
        ),
    )

    answer = AnswerSynthesizer(generator).synthesize(
        question="Why does Helen Haines support the bill?",
        evidence=[evidence],
    )

    assert answer.text == (
        "Helen Haines supports the bill because she says it helps children and parents. [1]\n\n"
        "Sources:\n"
        "[1] Helen Haines | 2025-02-12 | House of Representatives (Main) | "
        "source `282335_2025-02-12_hor_main_16_0_0_7_0` | chunk `chunk-haines`"
    )
    assert answer.citations[0].chunk_id == "chunk-haines"
    assert answer.citations[0].passage == evidence.chunk_text


def test_synthesis_rejects_a_claim_that_cites_unretrieved_evidence() -> None:
    """Prevent generated claims from linking to chunks absent from retrieval."""
    generator = StubAnswerGenerator(
        AnswerDraft(
            claims=(
                AnswerClaim(
                    text="The speaker supports the bill.",
                    evidence_chunk_ids=("invented-chunk",),
                ),
            ),
        ),
    )

    with pytest.raises(InvalidEvidenceReferenceError, match="was not retrieved"):
        AnswerSynthesizer(generator).synthesize(
            question="What position did the speaker take?",
            evidence=[_speech_chunk()],
        )


def test_synthesis_abstains_without_evidence() -> None:
    """Return a clear abstention instead of asking a model to answer without evidence."""
    generator = StubAnswerGenerator(AnswerDraft())

    answer = AnswerSynthesizer(generator).synthesize(
        question="What position did the speaker take?",
        evidence=[],
    )

    assert answer.abstained
    assert answer.citations == ()
    assert "No evidence passages were retrieved." in answer.text
    assert not generator.called


def test_synthesis_preserves_a_model_abstention() -> None:
    """Keep the generation adapter's reason when retrieved passages are insufficient."""
    generator = StubAnswerGenerator(
        AnswerDraft(
            abstained=True,
            abstention_reason="The passages do not identify the speaker's position.",
        ),
    )

    answer = AnswerSynthesizer(generator).synthesize(
        question="Did the speaker support the bill?",
        evidence=[_speech_chunk()],
    )

    assert answer.abstained
    assert answer.citations == ()
    assert answer.abstention_reason == "The passages do not identify the speaker's position."
    assert "The passages do not identify the speaker's position." in answer.text
