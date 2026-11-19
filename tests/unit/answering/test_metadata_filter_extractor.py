"""Verify structured date and speaker filter extraction for configured providers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import httpx
import pytest
from ollama import ChatResponse, Message
from openai import OpenAI

from ddd_policy_tracer.retrieval.filtering.metadata_filter_extractor import (
    METADATA_FILTER_SCHEMA,
    MetadataFilterExtractorConfig,
    OllamaMetadataFilterExtractor,
    OpenAIMetadataFilterExtractor,
)
from ddd_policy_tracer.retrieval.filtering.metadata_filters import (
    MetadataFilterExtraction,
    MetadataFilterExtractionError,
)

_FILTER_CONTENT = json.dumps(
    {
        "date_from": "2025-01-01",
        "date_to": "2025-01-31",
        "speaker_names": ["Tanya Plibersek"],
        "unsupported_filters": [],
        "clarification_question": None,
    },
)


class _OllamaClient:
    """Capture the Ollama request and return a fixed JSON-schema response."""

    def __init__(self, content: str) -> None:
        """Store the structured content to return."""
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
        """Return the requested extraction and capture its prompt."""
        self.request = {
            "model": model,
            "messages": messages,
            "format": format,
            "options": options,
            "stream": stream,
        }
        return ChatResponse(message=Message(role="assistant", content=self.content))


def _openai_client(
    content: str,
    capture_request: list[dict[str, Any]],
) -> OpenAI:
    """Build an OpenAI client backed by a local mocked HTTP response."""

    def respond(request: httpx.Request) -> httpx.Response:
        capture_request.append(json.loads(request.content))
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


def test_ollama_extractor_uses_schema_and_current_canberra_date() -> None:
    """Request structured constraints and provide date context for relative ranges."""
    client = _OllamaClient(_FILTER_CONTENT)
    extractor = OllamaMetadataFilterExtractor(
        MetadataFilterExtractorConfig(model="test-model"),
        client=client,
    )

    extracted = extractor.extract(
        "What did Tanya Plibersek say last month?",
        reference_date=date(2025, 2, 15),
    )

    assert extracted == MetadataFilterExtraction(
        date_from=date(2025, 1, 1),
        date_to=date(2025, 1, 31),
        speaker_names=("Tanya Plibersek",),
    )
    assert client.request is not None
    assert client.request["format"] == METADATA_FILTER_SCHEMA
    assert json.loads(client.request["messages"][1]["content"]) == {
        "current_canberra_date": "2025-02-15",
        "question": "What did Tanya Plibersek say last month?",
    }


def test_ollama_extractor_surfaces_invalid_provider_json() -> None:
    """Wrap malformed structured output as an explicit extraction failure."""
    extractor = OllamaMetadataFilterExtractor(
        MetadataFilterExtractorConfig(model="test-model"),
        client=_OllamaClient("not json"),
    )

    with pytest.raises(MetadataFilterExtractionError, match="invalid metadata-filter"):
        extractor.extract("Question?", reference_date=date(2025, 2, 15))


def test_openai_extractor_uses_strict_schema_and_matches_shared_contract() -> None:
    """Use strict JSON schema output and parse the same extraction contract."""
    requests: list[dict[str, Any]] = []
    extractor = OpenAIMetadataFilterExtractor(
        MetadataFilterExtractorConfig(model="test-model"),
        client=_openai_client(_FILTER_CONTENT, requests),
    )

    extracted = extractor.extract(
        "What did Tanya Plibersek say in January?",
        reference_date=date(2025, 2, 15),
    )

    assert extracted.speaker_names == ("Tanya Plibersek",)
    assert requests[0]["response_format"]["json_schema"]["strict"] is True
    assert requests[0]["response_format"]["json_schema"]["schema"] == METADATA_FILTER_SCHEMA
