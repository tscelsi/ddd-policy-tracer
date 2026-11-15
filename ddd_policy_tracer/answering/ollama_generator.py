"""Generate structured, evidence-linked answer drafts with Ollama."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ollama import ChatResponse, Client

from ddd_policy_tracer.answering.service_layer import (
    AnswerDraft,
    EvidenceChunk,
)
from ddd_policy_tracer.answering.structured_output import (
    ANSWER_SCHEMA,
    SYSTEM_PROMPT,
    parse_answer_draft,
)


class OllamaChatClient(Protocol):
    """Expose the non-streaming Ollama chat operation used by answer generation."""

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        format: dict[str, Any],
        options: Mapping[str, float],
        stream: bool,
    ) -> ChatResponse | Iterator[ChatResponse]:
        """Return one structured chat completion or a streaming response."""
        ...


@dataclass(frozen=True)
class OllamaAnswerGeneratorConfig:
    """Configure the local Ollama model used to draft grounded answer claims."""

    model: str = "llama3.1:8b"
    host: str | None = None
    request_timeout_seconds: float = 120.0
    temperature: float = 0.0

    def __post_init__(self) -> None:
        """Reject settings that cannot produce a deterministic usable request."""
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be greater than zero")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")


class OllamaAnswerGenerator:
    """Adapt Ollama chat completions to the structured answer-generation port."""

    def __init__(
        self,
        config: OllamaAnswerGeneratorConfig | None = None,
        *,
        client: OllamaChatClient | None = None,
    ) -> None:
        """Configure the model and reuse an injected client or create an Ollama client."""
        self.config = config or OllamaAnswerGeneratorConfig()
        self._client = (
            client
            if client is not None
            else Client(
                host=self.config.host,
                timeout=self.config.request_timeout_seconds,
            )
        )

    def generate(
        self,
        question: str,
        evidence: Sequence[EvidenceChunk],
    ) -> AnswerDraft:
        """Ask Ollama for claims linked to the supplied parliamentary passages."""
        if not question.strip():
            raise ValueError("question must not be empty")
        if not evidence:
            raise ValueError("evidence must not be empty")

        request = {
            "question": question,
            "evidence": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source_document_id": chunk.source_document_id,
                    "metadata": chunk.metadata,
                    "passage": chunk.chunk_text,
                }
                for chunk in evidence
            ],
        }
        response = self._client.chat(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(request, ensure_ascii=False),
                },
            ],
            format=ANSWER_SCHEMA,
            options={"temperature": self.config.temperature},
            stream=False,
        )
        if not isinstance(response, ChatResponse):
            raise TypeError("Ollama client returned a streaming response when stream=False")
        if response.message.content is None:
            raise ValueError("Ollama returned an empty answer response")
        return parse_answer_draft(response.message.content, provider="Ollama")
