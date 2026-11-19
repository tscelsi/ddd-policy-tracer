"""Tests for validating structured answer-generator responses."""

from __future__ import annotations

import json

import pytest

from ddd_policy_tracer.completion.synthesis.models import (
    AnswerClaim,
    AnswerDraft,
)


def test_parser_builds_pydantic_answer_models_and_normalizes_text() -> None:
    """Parse a provider response into validated answer-domain models."""
    content = json.dumps(
        {
            "claims": [{"text": "  A supported claim. ", "evidence_chunk_ids": ["chunk-1"]}],
            "abstained": False,
            "abstention_reason": None,
        },
    )

    draft = AnswerDraft.model_validate_json(content)

    assert draft == AnswerDraft(
        claims=(AnswerClaim(text="A supported claim.", evidence_chunk_ids=("chunk-1",)),),
        abstained=False,
        abstention_reason=None,
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"claims": [], "abstained": False, "abstention_reason": None, "unexpected": True},
        {"claims": [], "abstained": "false", "abstention_reason": None},
        {
            "claims": [{"text": "Claim.", "evidence_chunk_ids": ["chunk-1"], "extra": True}],
            "abstained": False,
            "abstention_reason": None,
        },
        {
            "claims": [{"text": "  ", "evidence_chunk_ids": ["chunk-1"]}],
            "abstained": False,
            "abstention_reason": None,
        },
        {
            "claims": [{"text": "Claim.", "evidence_chunk_ids": []}],
            "abstained": False,
            "abstention_reason": None,
        },
        {
            "claims": [{"text": "Claim.", "evidence_chunk_ids": [""]}],
            "abstained": False,
            "abstention_reason": None,
        },
        {"claims": [], "abstained": True, "abstention_reason": None},
        {
            "claims": [{"text": "Claim.", "evidence_chunk_ids": ["chunk-1"]}],
            "abstained": True,
            "abstention_reason": "Cannot answer.",
        },
        {
            "claims": [],
            "abstained": False,
            "abstention_reason": "Cannot answer.",
        },
    ],
)
def test_parser_rejects_responses_that_violate_the_answer_contract(
    payload: dict[str, object],
) -> None:
    """Report Pydantic validation failures with the provider's name."""
    with pytest.raises(ValueError, match="Provider answer response failed validation"):
        AnswerDraft.model_validate_json(json.dumps(payload))


def test_parser_rejects_invalid_json_with_provider_context() -> None:
    """Report malformed JSON as a provider response validation failure."""
    with pytest.raises(ValueError, match="Provider answer response failed validation"):
        AnswerDraft.model_validate_json("{")


def test_answer_draft_keeps_default_empty_value_for_internal_construction() -> None:
    """Allow the default draft for internal no-answer paths outside provider parsing."""
    assert AnswerDraft() == AnswerDraft(claims=(), abstained=False, abstention_reason=None)
