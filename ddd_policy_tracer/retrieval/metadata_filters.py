"""Domain contracts for extracting and applying parliamentary metadata filters."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

_CANBERRA = ZoneInfo("Australia/Canberra")


@dataclass(frozen=True)
class MetadataFilter:
    """Constrain retrieval to matching Hansard speakers and publication dates."""

    talker_ids: tuple[str, ...] = ()
    date_from: date | None = None
    date_to: date | None = None

    def __post_init__(self) -> None:
        """Reject contradictory date ranges and unusable speaker identifiers."""
        if self.date_from is not None and self.date_to is not None:
            if self.date_from > self.date_to:
                raise ValueError("date_from must not be later than date_to")
        if any(not talker_id.strip() for talker_id in self.talker_ids):
            raise ValueError("talker_ids must not contain empty values")
        object.__setattr__(self, "talker_ids", tuple(dict.fromkeys(self.talker_ids)))

    @property
    def is_empty(self) -> bool:
        """Report whether retrieval should remain unfiltered."""
        return not self.talker_ids and self.date_from is None and self.date_to is None


@dataclass(frozen=True)
class MetadataFilterExtraction:
    """Represent supported and unsupported constraints found in a user question."""

    date_from: date | None = None
    date_to: date | None = None
    speaker_names: tuple[str, ...] = ()
    unsupported_filters: tuple[str, ...] = ()
    clarification_question: str | None = None

    def __post_init__(self) -> None:
        """Reject an invalid extracted date range or blank requested names."""
        if self.date_from is not None and self.date_to is not None:
            if self.date_from > self.date_to:
                raise ValueError("date_from must not be later than date_to")
        if any(not name.strip() for name in self.speaker_names):
            raise ValueError("speaker_names must not contain empty values")


class MetadataFilterExtractor(Protocol):
    """Port for identifying filter intent without rewriting the original query."""

    def extract(
        self,
        question: str,
        *,
        reference_date: date,
    ) -> MetadataFilterExtraction:
        """Return date, speaker, and unsupported constraints in the question."""
        ...


class SpeakerNameResolver(Protocol):
    """Port for resolving a speaker name against the Hansard talker catalog."""

    async def resolve(self, speaker_names: Sequence[str]) -> SpeakerResolution:
        """Resolve names to talker IDs or request clarification."""
        ...


@dataclass(frozen=True)
class SpeakerResolution:
    """Return resolved speaker IDs or a question for an ambiguous name."""

    talker_ids: tuple[str, ...] = ()
    clarification_question: str | None = None


class MetadataFilterExtractionError(RuntimeError):
    """Signal unusable structured output from a metadata-filter provider."""


def parse_metadata_filter_extraction(content: str) -> MetadataFilterExtraction:
    """Validate the strict JSON contract returned by a metadata-filter provider."""
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Metadata filter response must be valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("Metadata filter response must be a JSON object")
    expected_keys = {
        "date_from",
        "date_to",
        "speaker_names",
        "unsupported_filters",
        "clarification_question",
    }
    if decoded.keys() != expected_keys:
        raise ValueError("Metadata filter response has missing or unexpected fields")

    date_from = _parse_date(decoded["date_from"], field_name="date_from")
    date_to = _parse_date(decoded["date_to"], field_name="date_to")
    speaker_names = _parse_string_list(decoded["speaker_names"], field_name="speaker_names")
    unsupported_filters = _parse_string_list(
        decoded["unsupported_filters"],
        field_name="unsupported_filters",
    )
    clarification_question = decoded["clarification_question"]
    if clarification_question is not None and not isinstance(clarification_question, str):
        raise ValueError("clarification_question must be a string or null")
    if clarification_question is not None and not clarification_question.strip():
        raise ValueError("clarification_question must not be blank")
    return MetadataFilterExtraction(
        date_from=date_from,
        date_to=date_to,
        speaker_names=tuple(speaker_names),
        unsupported_filters=tuple(unsupported_filters),
        clarification_question=clarification_question,
    )


def utc_published_at_bounds(
    *,
    date_from: date | None,
    date_to: date | None,
) -> tuple[str | None, str | None]:
    """Convert inclusive Canberra calendar dates to UTC ISO bounds for SQLite."""
    start = _canberra_midnight_utc(date_from) if date_from is not None else None
    end = _canberra_midnight_utc(date_to + timedelta(days=1)) if date_to is not None else None
    return _iso_utc(start), _iso_utc(end)


def _canberra_midnight_utc(day: date) -> datetime:
    """Map a local calendar midnight to its UTC instant."""
    return datetime.combine(day, time.min, tzinfo=_CANBERRA).astimezone(UTC)


def _iso_utc(value: datetime | None) -> str | None:
    """Serialize a UTC instant in the format stored for Hansard speech dates."""
    return value.isoformat() if value is not None else None


def _parse_date(value: object, *, field_name: str) -> date | None:
    """Parse an ISO calendar date without accepting arbitrary timestamp formats."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO date or null")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO calendar date") from exc


def _parse_string_list(value: object, *, field_name: str) -> list[str]:
    """Validate a JSON array containing only non-empty strings."""
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{field_name} must be an array of non-empty strings")
    return value
