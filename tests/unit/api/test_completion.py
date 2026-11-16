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
from pymongo.errors import AutoReconnect

from ddd_policy_tracer.answering.completion_service import (
    CompletionRun,
    RagCompletionService,
)
from ddd_policy_tracer.answering.metadata_filter_extraction import (
    MetadataFilterExtractionError,
    MetadataFilterIntent,
)
from ddd_policy_tracer.answering.ollama_generator import OllamaAnswerGenerator
from ddd_policy_tracer.answering.service_layer import (
    AnswerSynthesizer,
)
from ddd_policy_tracer.answering.speaker_resolution import SpeakerResolution
from ddd_policy_tracer.api.completion import get_completion_service, router
from ddd_policy_tracer.retrieval.service_layer import RetrievalFilters
from ddd_policy_tracer.retrieval.sqlite_retrieval import SQLiteChunkResult


class StubEmbeddingService:
    """Return a fixed embedding for API completion tests."""

    def __init__(self) -> None:
        """Prepare query tracking for the embedding boundary."""
        self.queries: list[str] = []

    def embed_one(
        self,
        text: str,
        prompt_name: str = "document",
        return_tensor: bool = False,
    ) -> list[float]:
        """Record the question and return the test vector."""
        self.queries.append(text)
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
        self.filters: RetrievalFilters | None = None
        self.calls = 0

    def search_hybrid(
        self,
        *,
        query: str,
        query_embedding: Sequence[float],
        limit: int = 10,
        candidate_limit: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> Sequence[SQLiteChunkResult]:
        """Return the known evidence and record the requested question."""
        self.query = query
        self.filters = filters
        self.calls += 1
        return self.results[:limit]


class StubFilterExtractor:
    """Return one controlled filter intent without contacting a model provider."""

    def __init__(self, intent: MetadataFilterIntent) -> None:
        """Retain the extraction result used by one completion test."""
        self.intent = intent
        self.questions: list[str] = []

    def extract_filters(self, question: str) -> MetadataFilterIntent:
        """Record the original question and return the controlled filter intent."""
        self.questions.append(question)
        return self.intent


class StubSpeakerResolver:
    """Resolve controlled speaker names to stable test talker identities."""

    def __init__(self, resolution: SpeakerResolution) -> None:
        """Retain the speaker resolution used by one completion test."""
        self.resolution = resolution
        self.names: list[tuple[str, ...]] = []

    def resolve(self, speaker_names: tuple[str, ...]) -> SpeakerResolution:
        """Record requested names and return the controlled resolution."""
        self.names.append(speaker_names)
        return self.resolution


class StubOllamaClient:
    """Return one structured answer response without a live Ollama server."""

    def __init__(self) -> None:
        """Track the original question sent to answer synthesis."""
        self.questions: list[str] = []

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
        self.questions.append(json.loads(messages[1]["content"])["question"])
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
        filter_extractor=StubFilterExtractor(MetadataFilterIntent()),
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
    assert "filter_extraction;dur=" in server_timing
    assert "speaker_resolution;dur=" in server_timing
    assert "embedding;dur=" in server_timing
    assert "hybrid_retrieval;dur=" in server_timing
    assert "generation;dur=" in server_timing
    assert "completion;dur=" in server_timing
    assert retriever.query == "Why does Helen Haines support the bill?"
    assert retriever.filters is None


def test_completion_passes_resolved_filters_without_rewriting_the_question() -> None:
    """Use speaker and date filters for retrieval while preserving the original question."""
    retriever = StubRetriever()
    embedding_service = StubEmbeddingService()
    question = "What did Helen Haines say in February 2025?"
    extractor = StubFilterExtractor(
        MetadataFilterIntent(
            start_date=date(2025, 2, 1),
            end_date=date(2025, 2, 28),
            speaker_names=("Helen Haines",),
        ),
    )
    resolver = StubSpeakerResolver(SpeakerResolution(talker_ids=("haines-id",)))
    answer_provider = StubOllamaClient()
    completion_service = RagCompletionService(
        retriever=retriever,
        embedding_service=embedding_service,
        synthesizer=AnswerSynthesizer(OllamaAnswerGenerator(client=answer_provider)),
        filter_extractor=extractor,
        speaker_resolver=resolver,
    )

    run = completion_service.complete(question)

    assert run.answer is not None
    assert extractor.questions == [question]
    assert resolver.names == [("Helen Haines",)]
    assert retriever.query == question
    assert embedding_service.queries == [question]
    assert answer_provider.questions == [question]
    assert retriever.filters == RetrievalFilters(
        talker_ids=("haines-id",),
        start_date=date(2025, 2, 1),
        end_date=date(2025, 2, 28),
    )


def test_empty_filtered_results_abstain_without_retrying_unfiltered() -> None:
    """Keep a valid empty result empty and let the normal abstention path answer."""
    retriever = StubRetriever()
    retriever.results = []
    question = "What did Helen Haines say in February 2025?"
    filters = RetrievalFilters(
        talker_ids=("haines-id",),
        start_date=date(2025, 2, 1),
        end_date=date(2025, 2, 28),
    )
    completion_service = RagCompletionService(
        retriever=retriever,
        embedding_service=StubEmbeddingService(),
        synthesizer=AnswerSynthesizer(OllamaAnswerGenerator(client=StubOllamaClient())),
        filter_extractor=StubFilterExtractor(
            MetadataFilterIntent(
                start_date=filters.start_date,
                end_date=filters.end_date,
                speaker_names=("Helen Haines",),
            ),
        ),
        speaker_resolver=StubSpeakerResolver(SpeakerResolution(talker_ids=filters.talker_ids)),
    )

    run = completion_service.complete(question)

    assert run.answer is not None
    assert run.answer.abstained
    assert run.answer.citations == ()
    assert retriever.calls == 1
    assert retriever.filters == filters


def test_completion_returns_clarification_without_embedding_or_retrieval() -> None:
    """Stop when the request includes a constraint the retriever cannot honor."""
    retriever = StubRetriever()
    embedding_service = StubEmbeddingService()
    completion_service = RagCompletionService(
        retriever=retriever,
        embedding_service=embedding_service,
        synthesizer=AnswerSynthesizer(OllamaAnswerGenerator(client=StubOllamaClient())),
        filter_extractor=StubFilterExtractor(
            MetadataFilterIntent(unsupported_constraints=("party",)),
        ),
    )
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_completion_service] = lambda: completion_service

    with TestClient(app) as client:
        response = client.post(
            "/api/completion",
            json={"question": "What did Labor members say about the bill?"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": None,
        "citations": [],
        "abstained": False,
        "abstention_reason": None,
        "clarification_required": True,
        "clarification_question": (
            "I can filter by date and speaker, but not by party. "
            "Could you ask without that constraint?"
        ),
    }
    assert embedding_service.queries == []
    assert retriever.calls == 0


def test_unknown_speaker_returns_clarification_without_retrieval() -> None:
    """Return a speaker clarification before embedding or searching the corpus."""
    retriever = StubRetriever()
    embedding_service = StubEmbeddingService()
    completion_service = RagCompletionService(
        retriever=retriever,
        embedding_service=embedding_service,
        synthesizer=AnswerSynthesizer(OllamaAnswerGenerator(client=StubOllamaClient())),
        filter_extractor=StubFilterExtractor(
            MetadataFilterIntent(speaker_names=("Unknown Speaker",)),
        ),
        speaker_resolver=StubSpeakerResolver(
            SpeakerResolution(
                clarification_question="Which Hansard speaker did you mean?",
            ),
        ),
    )

    run = completion_service.complete("What did Unknown Speaker say?")

    assert run.clarification_required
    assert run.answer is None
    assert run.clarification_question == "Which Hansard speaker did you mean?"
    assert embedding_service.queries == []
    assert retriever.calls == 0


def test_malformed_filter_output_returns_gateway_error_without_broadening_search() -> None:
    """Surface invalid provider constraints rather than silently searching unfiltered."""

    class InvalidFilterExtractor:
        """Raise the structured-output validation error for this completion."""

        def extract_filters(self, question: str) -> MetadataFilterIntent:
            """Reject malformed provider output."""
            raise MetadataFilterExtractionError("invalid structured filter response")

    retriever = StubRetriever()
    embedding_service = StubEmbeddingService()
    completion_service = RagCompletionService(
        retriever=retriever,
        embedding_service=embedding_service,
        synthesizer=AnswerSynthesizer(OllamaAnswerGenerator(client=StubOllamaClient())),
        filter_extractor=InvalidFilterExtractor(),
    )
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_completion_service] = lambda: completion_service

    with TestClient(app) as client:
        response = client.post("/api/completion", json={"question": "Question?"})

    assert response.status_code == 502
    assert response.json() == {
        "detail": "The answer generation provider returned invalid metadata filters.",
    }
    assert embedding_service.queries == []
    assert retriever.calls == 0


def test_completion_maps_openai_connection_errors_to_service_unavailable() -> None:
    """Return HTTP 503 when OpenAI cannot be reached."""

    class UnavailableCompletionService(RagCompletionService):
        """Raise the provider failure surfaced by the OpenAI SDK."""

        def complete(self, question: str, *, limit: int = 10) -> CompletionRun:
            """Raise a connection error without making a network request."""
            raise APIConnectionError(
                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            )

    completion_service = UnavailableCompletionService(
        retriever=StubRetriever(),
        embedding_service=StubEmbeddingService(),
        synthesizer=AnswerSynthesizer(OllamaAnswerGenerator(client=StubOllamaClient())),
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


def test_completion_maps_mongo_connectivity_errors_to_service_unavailable() -> None:
    """Distinguish a failed Hansard speaker lookup from a valid empty result."""

    class UnavailableCatalogCompletionService(RagCompletionService):
        """Raise the repository connectivity failure without contacting MongoDB."""

        def complete(self, question: str, *, limit: int = 10) -> CompletionRun:
            """Raise an operational Mongo connectivity error."""
            raise AutoReconnect("MongoDB is unavailable")

    completion_service = UnavailableCatalogCompletionService(
        retriever=StubRetriever(),
        embedding_service=StubEmbeddingService(),
        synthesizer=AnswerSynthesizer(OllamaAnswerGenerator(client=StubOllamaClient())),
    )
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_completion_service] = lambda: completion_service

    with TestClient(app) as client:
        response = client.post(
            "/api/completion",
            json={"question": "What did Helen Haines say?"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "The Hansard speaker catalog is unavailable."}
