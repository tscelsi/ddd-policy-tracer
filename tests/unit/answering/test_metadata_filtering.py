"""Tests for structured metadata filter validation and speaker-name resolution."""

from __future__ import annotations

import json
from datetime import date

import pytest

from ddd_policy_tracer.answering.metadata_filter_extraction import (
    MetadataFilterExtractionError,
    MetadataFilterIntent,
    parse_filter_intent,
)
from ddd_policy_tracer.answering.speaker_resolution import (
    SpeakerResolution,
    TalkerNameResolver,
)
from ddd_policy_tracer.ingestion.hansard.talker_repository import Talker


class StubTalkerCatalog:
    """Return a fixed talker catalog and count requests for lazy-load checks."""

    def __init__(self) -> None:
        """Prepare unique and ambiguous Hansard talker records."""
        self.calls = 0
        self.talkers = [
            Talker(
                id="haines-id",
                name="Haines, Helen MP",
                first_name="Helen",
                last_name="Haines",
                electorate="Indi",
                party="Independent",
            ),
            Talker(
                id="plibersek-id",
                name="Plibersek, Tanya Joan MP",
                first_name="Tanya",
                last_name="Plibersek",
                electorate="Sydney",
                party="ALP",
            ),
            Talker(
                id="smith-one",
                name="Smith, Alex",
                first_name="Alex",
                last_name="Smith",
                electorate="One",
                party="ALP",
            ),
            Talker(
                id="smith-two",
                name="Smith, Jamie",
                first_name="Jamie",
                last_name="Smith",
                electorate="Two",
                party="Liberal",
            ),
        ]

    def list_talkers(self) -> list[Talker]:
        """Return the catalog records and increment its access counter."""
        self.calls += 1
        return self.talkers


def test_filter_intent_parses_valid_dates_and_speaker_names() -> None:
    """Convert provider JSON to validated date and speaker intent."""
    intent = parse_filter_intent(
        json.dumps(
            {
                "start_date": "2025-02-01",
                "end_date": "2025-02-28",
                "speaker_names": [" Helen Haines "],
                "unsupported_constraints": [],
                "clarification_question": None,
            },
        ),
        provider="test provider",
    )

    assert intent == MetadataFilterIntent(
        start_date=date(2025, 2, 1),
        end_date=date(2025, 2, 28),
        speaker_names=("Helen Haines",),
    )


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        '{"start_date":"2025-02-31","end_date":null,"speaker_names":[],'
        '"unsupported_constraints":[],"clarification_question":null}',
        '{"start_date":null,"end_date":null,"speaker_names":[" "],'
        '"unsupported_constraints":[],"clarification_question":null}',
        '{"start_date":null,"end_date":null,"speaker_names":[],"unsupported_constraints":[]}',
    ],
)
def test_filter_intent_rejects_malformed_provider_responses(content: str) -> None:
    """Reject malformed structured output instead of treating it as no filters."""
    with pytest.raises(MetadataFilterExtractionError):
        parse_filter_intent(content, provider="test provider")


def test_speaker_resolver_supports_canonical_first_last_and_unique_surname_forms() -> None:
    """Resolve ordinary user name forms to stable catalog identities."""
    catalog = StubTalkerCatalog()
    resolver = TalkerNameResolver(catalog)

    canonical = resolver.resolve(("Haines, Helen MP",))
    first_last = resolver.resolve(("Tanya Plibersek",))
    unique_surname = resolver.resolve(("Haines",))

    assert canonical == SpeakerResolution(talker_ids=("haines-id",))
    assert first_last == SpeakerResolution(talker_ids=("plibersek-id",))
    assert unique_surname == SpeakerResolution(talker_ids=("haines-id",))
    assert catalog.calls == 1


def test_speaker_resolver_clarifies_ambiguous_and_unknown_names() -> None:
    """Avoid guessing when a surname is shared or absent from the talker catalog."""
    resolver = TalkerNameResolver(StubTalkerCatalog())

    ambiguous = resolver.resolve(("Smith",))
    unknown = resolver.resolve(("Not a Hansard speaker",))

    assert ambiguous.talker_ids == ()
    assert ambiguous.clarification_question == "Which speaker named Smith do you mean?"
    assert unknown.talker_ids == ()
    assert unknown.clarification_question is not None
    assert "couldn't find" in unknown.clarification_question
