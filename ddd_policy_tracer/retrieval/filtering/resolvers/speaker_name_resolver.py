"""Module for resolving speaker names against the Hansard talker catalog."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ddd_policy_tracer.ingestion.hansard.talker_repository import (
    AbstractTalkerRepository,
    Talker,
)

_WORD = re.compile(r"[^\W_]+", flags=re.UNICODE)


@dataclass(frozen=True)
class SpeakerResolution:
    """Return resolved speaker IDs or a question for an ambiguous name."""

    talker_ids: tuple[str, ...] = ()
    clarification_question: str | None = None


class AbstractSpeakerNameResolver(Protocol):
    """Port for resolving a speaker name against the Hansard talker catalog."""

    async def resolve(self, speaker_names: Sequence[str]) -> SpeakerResolution:
        """Resolve names to talker IDs or request clarification."""
        ...


class MongoTalkerNameResolver(AbstractSpeakerNameResolver):
    """Resolve canonical names against a lazily loaded, process-cached catalog."""

    def __init__(self, repository: AbstractTalkerRepository) -> None:
        """Accept the Mongo-backed talker repository port."""
        self._repository = repository
        self._talkers: tuple[Talker, ...] | None = None
        self._catalog_lock = asyncio.Lock()

    async def resolve(self, speaker_names: Sequence[str]) -> SpeakerResolution:
        """Return unique talker IDs, or a clarification for the first unresolved name."""
        if not speaker_names:
            return SpeakerResolution()
        talkers = await self._load_catalog()
        resolved_ids: list[str] = []
        for speaker_name in speaker_names:
            matches = {
                talker.id
                for talker in talkers
                if talker.id is not None and _matches_speaker(speaker_name, talker)
            }
            if len(matches) != 1:
                reason = "I couldn't find" if not matches else "I found multiple matches for"
                return SpeakerResolution(
                    clarification_question=(
                        f"{reason} the Hansard speaker “{speaker_name}”. "
                        "Could you provide a more specific name?"
                    ),
                )
            resolved_ids.extend(matches)
        return SpeakerResolution(talker_ids=tuple(dict.fromkeys(resolved_ids)))

    async def _load_catalog(self) -> tuple[Talker, ...]:
        """Load the catalog once, preserving a single concurrent Mongo request."""
        if self._talkers is not None:
            return self._talkers
        async with self._catalog_lock:
            if self._talkers is None:
                self._talkers = tuple(await self._repository.list_talkers())
        return self._talkers


def _matches_speaker(query: str, talker: Talker) -> bool:
    """Match exact canonical, first-last, or unique last-name forms."""
    query_tokens = _tokens(query)
    last_name_tokens = _tokens(talker.last_name)
    first_last_tokens = set(_tokens(f"{talker.first_name} {talker.last_name}"))
    canonical_tokens = set(_tokens(talker.name))
    if not query_tokens or not last_name_tokens:
        return False
    if len(query_tokens) == 1:
        return query_tokens[0] == last_name_tokens[-1]
    query_set = set(query_tokens)
    return query_set == canonical_tokens or (
        query_set.issubset(first_last_tokens) and last_name_tokens[-1] in query_set
    )


def _tokens(value: str) -> tuple[str, ...]:
    """Normalize a name into case-folded alphanumeric tokens."""
    return tuple(token.casefold() for token in _WORD.findall(value))
