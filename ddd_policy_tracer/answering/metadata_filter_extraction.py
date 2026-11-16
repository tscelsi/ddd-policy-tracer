"""Structured extraction of supported metadata constraints from parliamentary questions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Protocol
from zoneinfo import ZoneInfo

CANBERRA = ZoneInfo("Australia/Canberra")

FILTER_EXTRACTION_PROMPT = """\
Extract only retrieval constraints explicitly requested in the user's question.
Return inclusive calendar-date bounds, named speakers, explicitly requested unsupported
metadata constraints, and a clarification question when intent is unclear. Supported
constraints are dates and speakers only; other requested dimensions are unsupported.
Do not rewrite the question. Use null for absent dates and empty lists for absent names or
unsupported constraints.

Interpret dates in Australia/Canberra time. Use the supplied Canberra date as today.
A date without a year uses today's Canberra calendar year. "Last month" and "last year"
mean the previous complete calendar period. "Past N days" starts N days before today
and ends today, inclusive. Use ISO YYYY-MM-DD values and do not infer constraints from
the subject matter of an ordinary content question.
"""

FILTER_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "start_date": {"type": ["string", "null"]},
        "end_date": {"type": ["string", "null"]},
        "speaker_names": {"type": "array", "items": {"type": "string"}},
        "unsupported_constraints": {"type": "array", "items": {"type": "string"}},
        "clarification_question": {"type": ["string", "null"]},
    },
    "required": [
        "start_date",
        "end_date",
        "speaker_names",
        "unsupported_constraints",
        "clarification_question",
    ],
    "additionalProperties": False,
}


class MetadataFilterExtractor(Protocol):
    """Port for extracting structured metadata constraints without rewriting a question."""

    def extract_filters(self, question: str) -> MetadataFilterIntent:
        """Return date, speaker, unsupported, and clarification intent from a question."""
        ...


class MetadataFilterExtractionError(ValueError):
    """Signal malformed or unusable structured output from the extraction provider."""


@dataclass(frozen=True)
class MetadataFilterIntent:
    """Represent validated filter intent returned by a configured answer provider."""

    start_date: date | None = None
    end_date: date | None = None
    speaker_names: tuple[str, ...] = ()
    unsupported_constraints: tuple[str, ...] = ()
    clarification_question: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the extraction's domain-level intent."""
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.start_date > self.end_date
        ):
            raise ValueError("metadata filter start_date must not be after end_date")
        if any(not isinstance(name, str) or not name.strip() for name in self.speaker_names):
            raise ValueError("metadata filter speaker names must not be empty")
        if any(
            not isinstance(item, str) or not item.strip() for item in self.unsupported_constraints
        ):
            raise ValueError("unsupported metadata constraints must not be empty")
        if self.clarification_question is not None and not self.clarification_question.strip():
            raise ValueError("clarification_question must not be empty")
        object.__setattr__(
            self,
            "speaker_names",
            tuple(name.strip() for name in self.speaker_names),
        )
        object.__setattr__(
            self,
            "unsupported_constraints",
            tuple(item.strip() for item in self.unsupported_constraints),
        )
        if self.clarification_question is not None:
            object.__setattr__(self, "clarification_question", self.clarification_question.strip())


def current_canberra_date() -> date:
    """Return today's calendar date in the parliamentary timezone."""
    return datetime.now(CANBERRA).date()


def build_filter_request_content(question: str) -> str:
    """Serialize the unchanged question with its Canberra-date reference for extraction."""
    if not question.strip():
        raise ValueError("question must not be empty")
    return json.dumps(
        {
            "question": question,
            "canberra_today": current_canberra_date().isoformat(),
        },
        ensure_ascii=False,
    )


def parse_filter_intent(content: str, *, provider: str) -> MetadataFilterIntent:
    """Validate a provider's structured response before it reaches retrieval."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise MetadataFilterExtractionError(
            f"{provider} metadata filter response must contain valid JSON",
        ) from exc

    expected_keys = set(FILTER_EXTRACTION_SCHEMA["required"])
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise MetadataFilterExtractionError(
            f"{provider} metadata filter response has an invalid object shape",
        )

    start_date = _parse_date(payload["start_date"], provider=provider, field="start_date")
    end_date = _parse_date(payload["end_date"], provider=provider, field="end_date")
    speaker_names = _parse_nonempty_strings(
        payload["speaker_names"],
        provider=provider,
        field="speaker_names",
    )
    unsupported = _parse_nonempty_strings(
        payload["unsupported_constraints"],
        provider=provider,
        field="unsupported_constraints",
    )
    clarification = payload["clarification_question"]
    if clarification is not None and (
        not isinstance(clarification, str) or not clarification.strip()
    ):
        raise MetadataFilterExtractionError(
            f"{provider} metadata filter response has an invalid clarification",
        )

    try:
        return MetadataFilterIntent(
            start_date=start_date,
            end_date=end_date,
            speaker_names=speaker_names,
            unsupported_constraints=unsupported,
            clarification_question=clarification,
        )
    except ValueError as exc:
        raise MetadataFilterExtractionError(
            f"{provider} metadata filter response contains invalid constraints",
        ) from exc


def _parse_date(value: object, *, provider: str, field: str) -> date | None:
    """Convert one strictly formatted provider date into a calendar date."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise MetadataFilterExtractionError(
            f"{provider} metadata filter {field} must be a date or null",
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise MetadataFilterExtractionError(
            f"{provider} metadata filter {field} must use YYYY-MM-DD",
        ) from exc
    if parsed.isoformat() != value:
        raise MetadataFilterExtractionError(
            f"{provider} metadata filter {field} must use YYYY-MM-DD",
        )
    return parsed


def _parse_nonempty_strings(
    value: object,
    *,
    provider: str,
    field: str,
) -> tuple[str, ...]:
    """Validate an extraction field containing non-empty strings."""
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise MetadataFilterExtractionError(
            f"{provider} metadata filter {field} must contain non-empty strings",
        )
    return tuple(value)
