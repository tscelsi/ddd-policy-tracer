"""Provider adapters for structured metadata-filter extraction."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

from ollama import ChatResponse, Client
from openai import OpenAI

from ddd_policy_tracer.completion.filtering.metadata_filters import (
    METADATA_FILTER_SCHEMA,
    MetadataFilterExtraction,
    MetadataFilterExtractionError,
)

_SYSTEM_PROMPT = """Extract only metadata constraints the user requests for narrowing
parliamentary speech retrieval. Never rewrite the question.

Supported constraints:
- `date_from` and `date_to`: inclusive Australian/Canberra calendar dates. Use
  the supplied current date when a year is omitted. "Last month" and "last
  year" mean the previous complete calendar period; "past N days/months/years"
  means a rolling period ending today.
- `speaker_names`: people whose speeches the user explicitly asks to search.

Put explicit constraints on other metadata (such as party, house, chamber, bill,
or debate category) in `unsupported_filters`; do not silently omit them. Do not
treat a topic merely mentioned in the question as a metadata constraint. Return
an empty object of constraints for ordinary content questions. Use null for
missing dates and clarification_question unless the request itself is unclear.
"""


class OllamaMetadataChatClient(Protocol):
    """Expose the non-streaming Ollama chat operation for filter extraction."""

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        format: dict[str, Any],
        options: Mapping[str, float],
        stream: bool,
    ) -> ChatResponse | Iterator[ChatResponse]:
        """Return one structured extraction or a streaming response."""
        ...


@dataclass(frozen=True)
class MetadataFilterExtractorConfig:
    """Configure provider-neutral metadata extraction generation."""

    model: str
    host: str | None = None
    request_timeout_seconds: float = 120.0
    temperature: float = 0.0

    def __post_init__(self) -> None:
        """Reject extraction settings that cannot complete a deterministic request."""
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be greater than zero")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")


class AbstractMetadataFilterExtractor(Protocol):
    """Port for identifying filter intent without rewriting the original query."""

    def extract(
        self,
        question: str,
        *,
        reference_date: date,
    ) -> MetadataFilterExtraction:
        """Return date, speaker, and unsupported constraints in the question."""
        ...


class OllamaMetadataFilterExtractor(AbstractMetadataFilterExtractor):
    """Use Ollama structured output to detect dates and requested speakers."""

    def __init__(
        self,
        config: MetadataFilterExtractorConfig,
        *,
        client: OllamaMetadataChatClient | None = None,
    ) -> None:
        """Accept an injected Ollama client or configure the local default."""
        self.config = config
        self._client = client or Client(
            host=config.host,
            timeout=config.request_timeout_seconds,
        )

    def extract(self, question: str, *, reference_date: date) -> MetadataFilterExtraction:
        """Extract supported and unsupported retrieval constraints from a question."""
        response = self._client.chat(  # type: ignore[call-arg]
            model=self.config.model,
            messages=_messages(question, reference_date),
            format=METADATA_FILTER_SCHEMA,
            options={"temperature": self.config.temperature},
            stream=False,
        )
        if not isinstance(response, ChatResponse):
            raise MetadataFilterExtractionError(
                "Ollama returned a streaming metadata-filter response.",
            )
        if response.message.content is None:
            raise MetadataFilterExtractionError(
                "Ollama returned an empty metadata-filter response.",
            )
        return MetadataFilterExtraction.model_validate_json(response.message.content)


class OpenAIMetadataFilterExtractor(AbstractMetadataFilterExtractor):
    """Use OpenAI JSON-schema output to detect dates and requested speakers."""

    def __init__(
        self,
        config: MetadataFilterExtractorConfig,
        *,
        client: OpenAI | None = None,
    ) -> None:
        """Accept an injected OpenAI client or configure the environment-backed client."""
        self.config = config
        self._client = client or OpenAI(timeout=config.request_timeout_seconds)

    def extract(self, question: str, *, reference_date: date) -> MetadataFilterExtraction:
        """Extract supported and unsupported retrieval constraints from a question."""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=_messages(question, reference_date),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "parliamentary_metadata_filters",
                    "strict": True,
                    "schema": METADATA_FILTER_SCHEMA,
                },
            },
            temperature=self.config.temperature,
        )
        if not response.choices:
            raise MetadataFilterExtractionError(
                "OpenAI returned no metadata-filter choices.",
            )
        message = response.choices[0].message
        if message.refusal:
            raise MetadataFilterExtractionError(
                f"OpenAI refused metadata-filter extraction: {message.refusal}",
            )
        if message.content is None:
            raise MetadataFilterExtractionError(
                "OpenAI returned an empty metadata-filter response.",
            )
        return MetadataFilterExtraction.model_validate_json(message.content)


def _messages(question: str, reference_date: date) -> list[dict[str, str]]:
    """Build the provider prompt with the current Canberra date for relative terms."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {"current_canberra_date": reference_date.isoformat(), "question": question},
                ensure_ascii=False,
            ),
        },
    ]
