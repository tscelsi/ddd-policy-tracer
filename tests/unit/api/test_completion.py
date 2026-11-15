"""Tests for the end-to-end question completion API."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from ollama import ChatResponse, Message

from ddd_policy_tracer.answering.completion_service import RagCompletionService
from ddd_policy_tracer.answering.ollama_generator import OllamaAnswerGenerator
from ddd_policy_tracer.answering.service_layer import (
    AnswerSynthesizer,
)
from ddd_policy_tracer.api.completion import get_completion_service, router
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

    def search_hybrid(
        self,
        *,
        query: str,
        query_embedding: Sequence[float],
        limit: int = 10,
        candidate_limit: int | None = None,
    ) -> Sequence[SQLiteChunkResult]:
        """Return the known evidence and record the requested question."""
        self.query = query
        return self.results[:limit]


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
    assert retriever.query == "Why does Helen Haines support the bill?"
