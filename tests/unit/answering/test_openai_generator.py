"""Tests for the OpenAI adapter that generates evidence-linked answer drafts."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from openai import OpenAI

from ddd_policy_tracer.answering.openai_generator import (
    OpenAIAnswerGenerator,
    OpenAIAnswerGeneratorConfig,
)
from ddd_policy_tracer.answering.service_layer import AnswerClaim, AnswerDraft
from ddd_policy_tracer.retrieval.sqlite_retrieval import SQLiteChunkResult


def _openai_client(
    content: str,
    capture_request: Callable[[dict[str, Any]], None] | None = None,
) -> OpenAI:
    """Build an OpenAI client backed by a local HTTP response."""
    def respond(request: httpx.Request) -> httpx.Response:
        if capture_request is not None:
            capture_request(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    },
                ],
            },
        )

    return OpenAI(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )


def _speech_chunk() -> SQLiteChunkResult:
    """Build a parliamentary passage with identifying source metadata."""
    return SQLiteChunkResult(
        chunk_id="chunk-haines",
        source_id="hansard",
        source_document_id="282335_2025-02-12_hor_main_16_0_0_7_0",
        chunk_index=0,
        start_char=0,
        end_char=49,
        chunk_text="I support the bill because it helps children.",
        topics=None,
        metadata='{"doc_metadata":{"talker":{"name":"Helen Haines"}}}',
    )


def test_openai_generator_returns_claims_linked_to_supplied_evidence() -> None:
    """Request schema-constrained claims and map their IDs into the answer contract."""
    request: dict[str, Any] | None = None

    def capture_request(payload: dict[str, Any]) -> None:
        nonlocal request
        request = payload

    client = _openai_client(
        json.dumps(
            {
                "claims": [
                    {
                        "text": "Helen Haines says the bill helps children.",
                        "evidence_chunk_ids": ["chunk-haines"],
                    },
                ],
                "abstained": False,
                "abstention_reason": None,
            },
        ),
        capture_request,
    )
    generator = OpenAIAnswerGenerator(
        OpenAIAnswerGeneratorConfig(model="test-model"),
        client=client,
    )

    draft = generator.generate("Why does she support the bill?", [_speech_chunk()])

    assert draft == AnswerDraft(
        claims=(
            AnswerClaim(
                text="Helen Haines says the bill helps children.",
                evidence_chunk_ids=("chunk-haines",),
            ),
        ),
    )
    assert request is not None
    assert request["model"] == "test-model"
    assert request["response_format"]["type"] == "json_schema"
    request_data = json.loads(request["messages"][1]["content"])
    assert request_data["question"] == "Why does she support the bill?"
    assert request_data["evidence"][0]["chunk_id"] == "chunk-haines"
    assert request_data["evidence"][0]["passage"] == _speech_chunk().chunk_text


def test_openai_generator_preserves_an_explicit_abstention() -> None:
    """Return the model's abstention state and reason as a structured draft."""
    generator = OpenAIAnswerGenerator(
        client=_openai_client(
            json.dumps(
                {
                    "claims": [],
                    "abstained": True,
                    "abstention_reason": "The passages do not state a position.",
                },
            ),
        ),
    )

    draft = generator.generate("Did the speaker support it?", [_speech_chunk()])

    assert draft == AnswerDraft(
        abstained=True,
        abstention_reason="The passages do not state a position.",
    )


def test_openai_generator_rejects_malformed_structured_output() -> None:
    """Surface malformed model output rather than returning an invalid answer draft."""
    generator = OpenAIAnswerGenerator(
        client=_openai_client("not json"),
    )

    with pytest.raises(ValueError, match="valid JSON"):
        generator.generate("Question?", [_speech_chunk()])
