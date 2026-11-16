"""Verify structured metadata filter parsing and date-bound conversion."""

import json
from datetime import date

import pytest

from ddd_policy_tracer.retrieval.metadata_filters import (
    MetadataFilterExtraction,
    parse_metadata_filter_extraction,
    utc_published_at_bounds,
)


def test_parse_metadata_filter_extraction_returns_typed_values() -> None:
    """Convert provider JSON into dates, speaker names, and unsupported constraints."""
    extracted = parse_metadata_filter_extraction(
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
        ("not json", "valid JSON"),
        (
            '{"date_from":"2024-02-31","date_to":null,"speaker_names":[],"unsupported_filters":[],"clarification_question":null}',
            "valid ISO calendar date",
        ),
        (
            '{"date_from":null,"date_to":null,"speaker_names":[],"unsupported_filters":[]}',
            "missing or unexpected fields",
        ),
    ],
)
def test_parse_metadata_filter_extraction_rejects_malformed_content(
    content: str,
    message: str,
) -> None:
    """Raise an explicit error for invalid structured provider output."""
    with pytest.raises(ValueError, match=message):
        parse_metadata_filter_extraction(content)


def test_utc_published_at_bounds_cover_inclusive_canberra_dates() -> None:
    """Convert inclusive Australian calendar dates into UTC timestamp bounds."""
    start, end = utc_published_at_bounds(
        date_from=date(2025, 1, 1),
        date_to=date(2025, 1, 1),
    )

    assert start == "2024-12-31T13:00:00+00:00"
    assert end == "2025-01-01T13:00:00+00:00"


def test_utc_published_at_bounds_support_open_ended_ranges() -> None:
    """Leave either UTC bound unset for one-sided date requests."""
    assert utc_published_at_bounds(date_from=None, date_to=date(2024, 6, 30)) == (
        None,
        "2024-06-30T14:00:00+00:00",
    )
    assert utc_published_at_bounds(date_from=date(2024, 7, 1), date_to=None) == (
        "2024-06-30T14:00:00+00:00",
        None,
    )
