"""Domain contracts for extracting and applying parliamentary metadata filters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from pydantic import BaseModel, model_validator


class MetadataFilterExtractionError(RuntimeError):
    """Signal unusable structured output from a metadata-filter provider."""


@dataclass(frozen=True)
class MetadataFilter:
    """Constrain retrieval to matching Hansard speakers and publication dates."""

    talker_ids: tuple[str, ...] = ()
    date_from: date | None = None
    date_to: date | None = None
    bill_ids: tuple[str, ...] = ()

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


class MetadataFilterExtraction(BaseModel):
    """Represent supported and unsupported constraints found in a user question."""

    model_config = {"frozen": True}

    date_from: date | None = None
    date_to: date | None = None
    speaker_names: tuple[str, ...] = ()
    bill_names: tuple[str, ...] = ()
    unsupported_filters: tuple[str, ...] = ()
    clarification_question: str | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> MetadataFilterExtraction:
        """Reject an invalid extracted date range or blank requested values."""
        if self.date_from is not None and self.date_to is not None:
            if self.date_from > self.date_to:
                raise ValueError("date_from must not be later than date_to")
        if any(not name.strip() for name in self.speaker_names):
            raise ValueError("speaker_names must not contain empty values")
        if any(not bill_name.strip() for bill_name in self.bill_names):
            raise ValueError("bill_names must not contain empty values")
        if any(not value.strip() for value in self.unsupported_filters):
            raise ValueError("unsupported_filters must not contain empty values")
        if self.clarification_question is not None:
            if not self.clarification_question.strip():
                raise ValueError("clarification_question must not be blank")
        return self


METADATA_FILTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "date_from": {"type": ["string", "null"]},
        "date_to": {"type": ["string", "null"]},
        "speaker_names": {"type": "array", "items": {"type": "string"}},
        "bill_names": {"type": "array", "items": {"type": "string"}},
        "unsupported_filters": {"type": "array", "items": {"type": "string"}},
        "clarification_question": {"type": ["string", "null"]},
    },
    "required": [
        "date_from",
        "date_to",
        "speaker_names",
        "bill_names",
        "unsupported_filters",
        "clarification_question",
    ],
    "additionalProperties": False,
}
