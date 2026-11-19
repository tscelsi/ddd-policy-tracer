"""Verify structured metadata filter parsing and date-bound conversion."""

import json
from datetime import date

import pytest
from pydantic import ValidationError

from ddd_policy_tracer.completion.filtering.metadata_filters import (
    MetadataFilterExtraction,
)


def test_parse_metadata_filter_extraction_returns_typed_values() -> None:
    """Convert provider JSON into dates, speaker names, and unsupported constraints."""
    extracted = MetadataFilterExtraction.model_validate_json(
        json.dumps(
            {
                "date_from": "2024-01-01",
                "date_to": None,
                "speaker_names": ["Tanya Plibersek"],
                "unsupported_filters": [],
                "clarification_question": None,
            },
        ),
    )

    assert extracted == MetadataFilterExtraction(
        date_from=date(2024, 1, 1),
        date_to=None,
        speaker_names=("Tanya Plibersek",),
        unsupported_filters=(),
        clarification_question=None,
    )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "Invalid JSON: expected ident at line 1 column 2"),
        (
            '{"date_from":"2024-02-31","date_to":null,"speaker_names":[],"unsupported_filters":[],"clarification_question":null}',
            "Input should be a valid date or datetime, day value is outside expected range",
        ),
    ],
)
def test_parse_metadata_filter_extraction_rejects_malformed_content(
    content: str,
    message: str,
) -> None:
    """Raise an explicit error for invalid structured provider output."""
    with pytest.raises(ValidationError) as e:
        MetadataFilterExtraction.model_validate_json(content)
    assert e.value.errors()[0]["msg"] == message
