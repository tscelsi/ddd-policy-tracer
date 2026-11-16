"""Tests for the end-to-end question completion API."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from ollama import ChatResponse, Message
from openai import APIConnectionError

from ddd_policy_tracer.answering.completion_service import (
    CompletionRun,
    RagCompletionService,
)
from ddd_policy_tracer.answering.ollama_generator import OllamaAnswerGenerator
from ddd_policy_tracer.answering.service_layer import (
    AnswerSynthesizer,
)
from ddd_policy_tracer.api.completion import get_completion_service, router
from ddd_policy_tracer.retrieval.metadata_filters import (
    MetadataFilter,
    MetadataFilterExtraction,
    SpeakerResolution,
)
from ddd_policy_tracer.retrieval.sqlite_retrieval import SQLiteChunkResult


class StubEmbeddingService:
    """Return a fixed embedding for API completion tests."""

    def embed_one(
        self,
        text: str,
        prompt_name: str = "document",
        return_tensor: bool = False,
    ) -> list[float]:
        """Return the test vector for a query."""
        return [0.0, 1.0, 0.0]


class StubRetriever:
    """Return one ranked speech passage to the completion use case."""

    def __init__(self) -> None:
        """Prepare the retrieved evidence passage."""
        self.results = [
            SQLiteChunkResult(
                chunk_id="chunk-haines",
                source_id="hansard",
                source_document_id="282335_2025-02-12_hor_main_16_0_0_7_0",
                chunk_index=0,
                start_char=0,
                end_char=49,
                chunk_text="I support the bill because it helps children.",
                topics=None,
                metadata='{"doc_metadata":{"talker":{"name":"Helen Haines"}}}',
            ),
        ]
        self.query: str | None = None
        self.filters: MetadataFilter | None = None

    def search_hybrid(
        self,
        *,
        query: str,
        query_embedding: Sequence[float],
        limit: int = 10,
        candidate_limit: int | None = None,
        filters: MetadataFilter | None = None,
    ) -> Sequence[SQLiteChunkResult]:
        """Return the known evidence and record the requested question."""
        self.query = query
        self.filters = filters
        return self.results[:limit]


class StubMetadataFilterExtractor:
    """Return one deterministic filter extraction for a completion test."""

    def __init__(self, result: MetadataFilterExtraction | None = None) -> None:
        """Store the structured extraction to return."""
        self.result = result or MetadataFilterExtraction()

    def extract(
        self,
        question: str,
        *,
        reference_date: date,
    ) -> MetadataFilterExtraction:
        """Return the configured filter extraction."""
        return self.result


class StubSpeakerNameResolver:
    """Return deterministic speaker identities without requiring MongoDB."""

    def __init__(self, result: SpeakerResolution | None = None) -> None:
        """Store the resolution used by the completion test."""
        self.result = result or SpeakerResolution()
        self.names: Sequence[str] = ()

    async def resolve(self, speaker_names: Sequence[str]) -> SpeakerResolution:
        """Record requested names and return the configured resolution."""
        self.names = speaker_names
        return self.result


class StubOllamaClient:
    """Return one structured answer response without a live Ollama server."""

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        format: dict[str, Any],
        options: Mapping[str, float],
        stream: bool,
    ) -> ChatResponse:
        """Return a claim that cites the retrieved speech chunk."""
        return ChatResponse(
            message=Message(
                role="assistant",
                content=json.dumps(
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
            ),
        )


def test_completion_runs_retrieval_and_returns_cited_answer() -> None:
    """Return the answer and citation after hybrid retrieval and synthesis."""
    retriever = StubRetriever()
    completion_service = RagCompletionService(
        retriever=retriever,
        embedding_service=StubEmbeddingService(),
        synthesizer=AnswerSynthesizer(OllamaAnswerGenerator(client=StubOllamaClient())),
        metadata_filter_extractor=StubMetadataFilterExtractor(
            MetadataFilterExtraction(
                date_from=date(2025, 2, 1),
                date_to=date(2025, 2, 28),
                speaker_names=("Helen Haines",),
            ),
        ),
        speaker_name_resolver=StubSpeakerNameResolver(
            SpeakerResolution(talker_ids=("talker-haines",)),
        ),
    )
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_completion_service] = lambda: completion_service

    with TestClient(app) as client:
        response = client.post(
            "/api/completion",
            json={"question": "Why does Helen Haines support the bill?"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": (
            "Helen Haines says the bill helps children. [1]\n\n"
            "Sources:\n"
            "[1] Helen Haines | 2025-02-12 | House of Representatives (Main) | "
            "source `282335_2025-02-12_hor_main_16_0_0_7_0` | chunk `chunk-haines`"
        ),
        "abstained": False,
        "abstention_reason": None,
        "clarification_required": False,
        "clarification_question": None,
        "citations": [
            {
                "number": 1,
                "chunk_id": "chunk-haines",
                "source_document_id": "282335_2025-02-12_hor_main_16_0_0_7_0",
                "passage": "I support the bill because it helps children.",
                "speaker": "Helen Haines",
                "date": "2025-02-12",
                "house": "House of Representatives",
                "chamber": "Main",
            },
        ],
    }
    server_timing = response.headers["server-timing"]
    assert "embedding;dur=" in server_timing
    assert "metadata_filter;dur=" in server_timing
    assert "hybrid_retrieval;dur=" in server_timing
    assert "generation;dur=" in server_timing
    assert "completion;dur=" in server_timing
    assert retriever.query == "Why does Helen Haines support the bill?"
    assert retriever.filters == MetadataFilter(
        talker_ids=("talker-haines",),
        date_from=date(2025, 2, 1),
        date_to=date(2025, 2, 28),
    )


def test_completion_maps_openai_connection_errors_to_service_unavailable() -> None:
    """Return HTTP 503 when OpenAI cannot be reached."""

    class UnavailableCompletionService(RagCompletionService):
        """Raise the provider failure surfaced by the OpenAI SDK."""

        async def complete(self, question: str, *, limit: int = 10) -> CompletionRun:
            """Raise a connection error without making a network request."""
            raise APIConnectionError(
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            )

    completion_service = UnavailableCompletionService(
        retriever=StubRetriever(),
        embedding_service=StubEmbeddingService(),
        synthesizer=AnswerSynthesizer(OllamaAnswerGenerator(client=StubOllamaClient())),
        metadata_filter_extractor=StubMetadataFilterExtractor(),
        speaker_name_resolver=StubSpeakerNameResolver(),
    )
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_completion_service] = lambda: completion_service

    with TestClient(app) as client:
        response = client.post(
            "/api/completion",
            json={"question": "Why does Helen Haines support the bill?"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "The answer generation provider is unavailable."}


def test_completion_returns_clarification_without_retrieval() -> None:
    """Ask the caller to clarify unsupported metadata instead of dropping the filter."""
    retriever = StubRetriever()
    completion_service = RagCompletionService(
        retriever=retriever,
        embedding_service=StubEmbeddingService(),
        synthesizer=AnswerSynthesizer(OllamaAnswerGenerator(client=StubOllamaClient())),
        metadata_filter_extractor=StubMetadataFilterExtractor(
            MetadataFilterExtraction(unsupported_filters=("party",)),
        ),
        speaker_name_resolver=StubSpeakerNameResolver(),
    )
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_completion_service] = lambda: completion_service

    with TestClient(app) as client:
        response = client.post(
            "/api/completion",
            json={"question": "Find speeches by the Labor party."},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": None,
        "citations": [],
        "abstained": False,
        "abstention_reason": None,
        "clarification_required": True,
        "clarification_question": (
            "I can currently narrow Hansard searches by speaker and date, "
            "but not by party. Could you rephrase your request?"
        ),
    }
    assert retriever.query is None


def test_completion_returns_speaker_clarification_without_retrieval() -> None:
    """Do not search broadly when a requested speaker could not be resolved."""
    retriever = StubRetriever()
    completion_service = RagCompletionService(
        retriever=retriever,
        embedding_service=StubEmbeddingService(),
        synthesizer=AnswerSynthesizer(OllamaAnswerGenerator(client=StubOllamaClient())),
        metadata_filter_extractor=StubMetadataFilterExtractor(
            MetadataFilterExtraction(speaker_names=("Smith",)),
        ),
        speaker_name_resolver=StubSpeakerNameResolver(
            SpeakerResolution(
                clarification_question="Which Smith did you mean?",
            ),
        ),
    )
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_completion_service] = lambda: completion_service

    with TestClient(app) as client:
        response = client.post(
            "/api/completion",
            json={"question": "What did Smith say?"},
        )

    assert response.status_code == 200
    assert response.json()["clarification_required"] is True
    assert response.json()["clarification_question"] == "Which Smith did you mean?"
    assert retriever.query is None
