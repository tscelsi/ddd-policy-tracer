"""Unit tests for Ollama-backed claim extraction behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ddd_policy_tracer.analysis.chunks.chunking_models import DocumentChunk
from ddd_policy_tracer.analysis.claims import (
    OllamaClaimExtractor,
    OllamaClaimExtractorConfig,
)


class StubResponse:
    """Represent one deterministic HTTP response stub."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Initialize response with static JSON payload data."""
        self._payload = payload

    def raise_for_status(self) -> None:
        """No-op status behavior for successful response stubs."""

    def json(self) -> dict[str, Any]:
        """Return response payload object for extractor parsing."""
        return self._payload


@dataclass
class StubHttpClient:
    """Capture outgoing request payloads and return static responses."""

    response_payload: dict[str, Any]
    seen_payloads: list[dict[str, Any]]

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> StubResponse:
        """Record request payload and return configured response stub."""
        self.seen_payloads.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
            },
        )
        return StubResponse(self.response_payload)


def _chunk(text: str) -> DocumentChunk:
    """Build one representative chunk fixture for extractor tests."""
    return DocumentChunk(
        chunk_id="chunk_1",
        source_id="australia_institute",
        source_document_id="https://example.org/report-1",
        document_checksum="checksum-1",
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        chunk_text=text,
    )


def test_ollama_extractor_builds_claim_candidates_from_substrings() -> None:
    """Map valid Ollama claims to chunk offsets and candidate fields."""
    chunk = _chunk(
        "Government should ban new coal projects. "
        "Policy should reduce emissions by 4.9%.",
    )
    client = StubHttpClient(
        response_payload={
            "message": {
                "content": (
                    '{"claims": ['
                    '"Government should ban new coal projects.", '
                    '"Policy should reduce emissions by 4.9%."'
                    "]}"
                ),
            },
        },
        seen_payloads=[],
    )
    extractor = OllamaClaimExtractor(
        config=OllamaClaimExtractorConfig(model="llama3.1:8b"),
        http_client=client,
    )

    claims = extractor.extract(chunk=chunk)

    assert len(claims) == 2
    assert claims[0].evidence_text == "Government should ban new coal projects."
    assert claims[1].evidence_text == "Policy should reduce emissions by 4.9%."
    assert claims[0].extractor_version == "ollama-v1"
    assert len(client.seen_payloads) == 1
    assert client.seen_payloads[0]["json"]["model"] == "llama3.1:8b"
    assert client.seen_payloads[0]["json"]["stream"] is False
    assert client.seen_payloads[0]["json"]["format"] == "json"


def test_ollama_extractor_skips_claims_not_found_in_chunk_text() -> None:
    """Drop model claims that are not exact matches within chunk text."""
    chunk = _chunk("Government should ban new coal projects.")
    client = StubHttpClient(
        response_payload={
            "message": {
                "content": '{"claims": ["Government should ban coal immediately."]}',
            },
        },
        seen_payloads=[],
    )
    extractor = OllamaClaimExtractor(http_client=client)

    claims = extractor.extract(chunk=chunk)

    assert claims == []


def test_ollama_extractor_raises_for_invalid_response_format() -> None:
    """Raise clear errors when Ollama response payload is malformed."""
    chunk = _chunk("Government should ban new coal projects.")
    extractor = OllamaClaimExtractor(
        http_client=StubHttpClient(
            response_payload={"unexpected": {"content": "{}"}},
            seen_payloads=[],
        ),
    )

    with pytest.raises(ValueError, match="Invalid Ollama response format"):
        extractor.extract(chunk=chunk)
