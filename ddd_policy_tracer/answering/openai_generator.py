"""Generate structured, evidence-linked answer drafts with OpenAI."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from openai import OpenAI

from ddd_policy_tracer.answering.metadata_filter_extraction import (
    FILTER_EXTRACTION_PROMPT,
    FILTER_EXTRACTION_SCHEMA,
    MetadataFilterExtractionError,
    MetadataFilterIntent,
    build_filter_request_content,
    parse_filter_intent,
)
from ddd_policy_tracer.answering.service_layer import (
    AnswerDraft,
    EvidenceChunk,
)
from ddd_policy_tracer.answering.structured_output import (
    ANSWER_SCHEMA,
    SYSTEM_PROMPT,
    parse_answer_draft,
)


@dataclass(frozen=True)
class OpenAIAnswerGeneratorConfig:
    """Configure the OpenAI model used to draft grounded answer claims."""

    model: str = "gpt-6-luna"
    request_timeout_seconds: float = 120.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        """Reject settings that cannot produce a deterministic usable request."""
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be greater than zero")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")


class OpenAIAnswerGenerator:
    """Adapt OpenAI chat completions to the structured answer-generation port."""

    def __init__(
        self,
        config: OpenAIAnswerGeneratorConfig | None = None,
        *,
        client: OpenAI | None = None,
    ) -> None:
        """Use an injected OpenAI client or create one from its environment configuration."""
        self.config = config or OpenAIAnswerGeneratorConfig()
        self._client = (
            client if client is not None else OpenAI(timeout=self.config.request_timeout_seconds)
        )

    def generate(
        self,
        question: str,
        evidence: Sequence[EvidenceChunk],
    ) -> AnswerDraft:
        """Ask OpenAI for claims linked to the supplied parliamentary passages."""
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
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(request, ensure_ascii=False),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "parliamentary_answer",
                    "strict": True,
                    "schema": ANSWER_SCHEMA,
                },
            },
            temperature=self.config.temperature,
        )
        if not response.choices:
            raise ValueError("OpenAI returned no answer choices")
        message = response.choices[0].message
        if message.refusal:
            raise ValueError(f"OpenAI refused to generate an answer: {message.refusal}")
        if message.content is None:
            raise ValueError("OpenAI returned an empty answer response")
        return parse_answer_draft(message.content, provider="OpenAI")

    def extract_filters(self, question: str) -> MetadataFilterIntent:
        """Extract only supported metadata constraints using the configured OpenAI model."""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": FILTER_EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": build_filter_request_content(question),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "parliamentary_metadata_filters",
                    "strict": True,
                    "schema": FILTER_EXTRACTION_SCHEMA,
                },
            },
            temperature=0,
        )
        if not response.choices:
            raise MetadataFilterExtractionError("OpenAI returned no metadata filter choices")
        message = response.choices[0].message
        if message.refusal:
            raise MetadataFilterExtractionError(
                f"OpenAI refused to extract metadata filters: {message.refusal}",
            )
        if message.content is None:
            raise MetadataFilterExtractionError("OpenAI returned an empty metadata filter response")
        return parse_filter_intent(message.content, provider="OpenAI")
