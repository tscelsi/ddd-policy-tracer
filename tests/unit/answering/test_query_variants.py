"""Tests for structured retrieval-query drafts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ddd_policy_tracer.completion.evaluation.query_variants import (
    EvidencePassage,
    OpenAIQueryVariantDrafter,
    parse_query_variants,
)


def test_query_parser_requires_two_to_three_distinct_nonempty_variants() -> None:
    """Reject malformed or duplicate provider output before a human sees it."""
    assert parse_query_variants(
        '{"queries":["What did Lee say about housing?",'
        '"Which housing measures did Lee describe?"]}',
        provider="test",
    ) == (
        "What did Lee say about housing?",
        "Which housing measures did Lee describe?",
    )

    with pytest.raises(ValueError, match="duplicate"):
        parse_query_variants(
            '{"queries":["What happened?"," what happened "]}',
            provider="test",
        )


def test_openai_drafter_sends_only_selected_evidence_and_reports_provider() -> None:
    """Ground structured query generation in selected passages and their metadata."""
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"queries":["What did Lee say about housing?",'
                    '"Which housing measures did Lee describe?"]}',
                    refusal=None,
                ),
            ),
        ],
    )
    requests: list[dict[str, object]] = []

    def create(**request: object) -> SimpleNamespace:
        """Capture the model request and return a structured response."""
        requests.append(request)
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
    )
    drafter = OpenAIQueryVariantDrafter(model="selected-model", client=client)  # type: ignore[arg-type]
    evidence = [
        EvidencePassage(
            chunk_id="chunk-1",
            chunk_text="Lee described new social housing investment.",
            source_document_id="speech-1",
            metadata={"speaker_name": "Lee", "published_at": "2025-01-01"},
        ),
    ]

    draft = drafter.draft(evidence)
    user_payload = json.loads(requests[0]["messages"][1]["content"])  # type: ignore[index]

    assert draft.provider == "openai"
    assert draft.model == "selected-model"
    assert len(draft.queries) == 2
    assert user_payload["selected_evidence"][0]["chunk_id"] == "chunk-1"
    assert user_payload["selected_evidence"][0]["metadata"]["speaker_name"] == "Lee"
    assert requests[0]["response_format"]["json_schema"]["strict"] is True  # type: ignore[index]


def test_openai_drafter_reports_refusal_and_empty_provider_responses() -> None:
    """Surface provider refusals and unavailable responses instead of saving cases."""
    refusal_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=None, refusal="cannot comply"),
                        ),
                    ],
                ),
            ),
        ),
    )
    drafter = OpenAIQueryVariantDrafter(client=refusal_client)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="refused"):
        drafter.draft(
            [EvidencePassage("chunk-1", "Some passage.", "speech-1", {})],
        )

    empty_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: SimpleNamespace(choices=[]),
            ),
        ),
    )
    empty_drafter = OpenAIQueryVariantDrafter(client=empty_client)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no query-variant choices"):
        empty_drafter.draft(
            [EvidencePassage("chunk-1", "Some passage.", "speech-1", {})],
        )
