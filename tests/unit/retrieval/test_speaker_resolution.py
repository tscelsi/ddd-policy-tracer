"""Verify unique deterministic mapping from names to Hansard talker IDs."""

import pytest

from ddd_policy_tracer.ingestion.hansard.talker_repository import Talker
from ddd_policy_tracer.retrieval.speaker_resolution import MongoTalkerNameResolver


class _TalkerRepository:
    """Provide an in-memory catalog through the talker repository port."""

    def __init__(self) -> None:
        """Create a catalog and count its loads."""
        self.calls = 0

    async def list_talkers(self) -> list[Talker]:
        """Return the catalog used by deterministic matching tests."""
        self.calls += 1
        return [
            Talker(
                id="talker-1",
                name="Plibersek, Tanya Joan MP",
                first_name="Tanya Joan",
                last_name="Plibersek",
                electorate="Sydney",
                party="ALP",
            ),
            Talker(
                id="talker-2",
                name="Smith, Alex MP",
                first_name="Alex",
                last_name="Smith",
                electorate=None,
                party="IND",
            ),
            Talker(
                id="talker-3",
                name="Smith, Jordan MP",
                first_name="Jordan",
                last_name="Smith",
                electorate=None,
                party="IND",
            ),
        ]


@pytest.mark.asyncio
async def test_resolver_matches_name_forms_and_caches_catalog() -> None:
    """Resolve canonical, first-last, and unique last names to stable IDs."""
    repository = _TalkerRepository()
    resolver = MongoTalkerNameResolver(repository)

    first = await resolver.resolve(["Tanya Plibersek", "Alex Smith", "Plibersek"])
    second = await resolver.resolve(["Tanya Plibersek"])

    assert first.talker_ids == ("talker-1", "talker-2")
    assert first.clarification_question is None
    assert second.talker_ids == ("talker-1",)
    assert repository.calls == 1


@pytest.mark.asyncio
async def test_resolver_requests_clarification_for_ambiguous_or_unknown_names() -> None:
    """Do not guess a speaker ID when a name cannot be uniquely resolved."""
    resolver = MongoTalkerNameResolver(_TalkerRepository())

    ambiguous = await resolver.resolve(["Smith"])
    unknown = await resolver.resolve(["Unknown Member"])

    assert ambiguous.talker_ids == ()
    assert "Smith" in (ambiguous.clarification_question or "")
    assert unknown.talker_ids == ()
    assert "Unknown Member" in (unknown.clarification_question or "")


@pytest.mark.asyncio
async def test_resolver_empty_names_does_not_load_catalog() -> None:
    """Skip Mongo access when the filter extraction contains no speaker names."""
    repository = _TalkerRepository()
    resolver = MongoTalkerNameResolver(repository)

    result = await resolver.resolve([])

    assert result.talker_ids == ()
    assert result.clarification_question is None
    assert repository.calls == 0
