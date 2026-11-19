"""Tests for the Ollama adapter that generates evidence-linked answer drafts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from ollama import ChatResponse, Message

from ddd_policy_tracer.completion.retrieval.sqlite_retrieval import SQLiteChunkResult
from ddd_policy_tracer.completion.synthesis.answer_synthesizer import (
    AnswerSynthesizer,
)
from ddd_policy_tracer.completion.synthesis.generators.ollama_generator import (
    OllamaAnswerGenerator,
    OllamaAnswerGeneratorConfig,
)
from ddd_policy_tracer.completion.synthesis.models import AnswerClaim, AnswerDraft


class StubOllamaClient:
    """Return a fixed Ollama response and retain the submitted chat request."""

    def __init__(self, content: str) -> None:
        """Prepare a fixed response for an adapter test."""
        self.content = content
        self.request: dict[str, Any] | None = None

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        format: dict[str, Any],
        options: Mapping[str, float],
        stream: bool,
    ) -> ChatResponse:
        """Record the request and return the configured structured response."""
        self.request = {
            "model": model,
            "messages": messages,
            "format": format,
            "options": options,
            "stream": stream,
        }
        return ChatResponse(message=Message(role="assistant", content=self.content))


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


def test_ollama_generator_returns_claims_linked_to_supplied_evidence() -> None:
    """Ask Ollama for structured, evidence-linked claims and parse its response."""
    client = StubOllamaClient(
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
    )
    generator = OllamaAnswerGenerator(
        OllamaAnswerGeneratorConfig(model="test-model"),
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
    assert client.request is not None
    assert client.request["model"] == "test-model"
    assert client.request["stream"] is False
    assert client.request["format"]["type"] == "object"
    user_message = client.request["messages"][1]
    request_data = json.loads(user_message["content"])
    assert request_data["question"] == "Why does she support the bill?"
    assert request_data["evidence"][0]["chunk_id"] == "chunk-haines"
    assert request_data["evidence"][0]["passage"] == _speech_chunk().chunk_text


def test_ollama_generator_preserves_an_explicit_abstention() -> None:
    """Return the model's abstention state and reason as a structured draft."""
    client = StubOllamaClient(
        json.dumps(
            {
                "claims": [],
                "abstained": True,
                "abstention_reason": "The passages do not state a position.",
            },
        ),
    )
    generator = OllamaAnswerGenerator(client=client)

    draft = generator.generate("Did the speaker support it?", [_speech_chunk()])

    assert draft.claims == ()
    assert draft.abstained
    assert draft.abstention_reason == "The passages do not state a position."


def test_ollama_generator_rejects_malformed_structured_output() -> None:
    """Surface invalid model output instead of silently manufacturing a draft."""
    generator = OllamaAnswerGenerator(client=StubOllamaClient("not json"))

    with pytest.raises(ValueError, match="valid JSON"):
        generator.generate("Question?", [_speech_chunk()])


def test_ollama_generator_produces_an_answer_with_resolved_citations() -> None:
    """Pass generated chunk references through to the source-backed answer renderer."""
    client = StubOllamaClient(
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
    )
    answer = AnswerSynthesizer(OllamaAnswerGenerator(client=client)).synthesize(
        question="Why does Helen Haines support the bill?",
        evidence=[_speech_chunk()],
    )

    assert not answer.abstained
    assert answer.text.startswith("Helen Haines says the bill helps children. [1]")
    assert answer.citations[0].chunk_id == "chunk-haines"
    assert "`282335_2025-02-12_hor_main_16_0_0_7_0`" in answer.text
