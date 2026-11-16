"""Resolve user-facing speaker names to stable Hansard talker identities."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

_NAME_TOKEN = re.compile(r"[^\w]+", flags=re.UNICODE)


class HansardSpeaker(Protocol):
    """Expose the identity and names needed to resolve one parliamentary speaker."""

    id: str | None
    name: str
    first_name: str
    last_name: str


class TalkerCatalog(Protocol):
    """Port for the Hansard catalog used to resolve parliamentary speakers."""

    def list_talkers(self) -> Sequence[HansardSpeaker]:
        """Return the catalog's available Hansard speakers."""
        ...


class SpeakerResolver(Protocol):
    """Port for deterministic user-name to talker-identity resolution."""

    def resolve(self, speaker_names: tuple[str, ...]) -> SpeakerResolution:
        """Resolve every requested speaker or return a clarification question."""
        ...


@dataclass(frozen=True)
class SpeakerResolution:
    """Return stable speaker identities or the clarification needed to disambiguate."""

    talker_ids: tuple[str, ...] = ()
    clarification_question: str | None = None

    def __post_init__(self) -> None:
        """Ensure a speaker lookup returns identities or a usable clarification."""
        if any(
            not isinstance(talker_id, str) or not talker_id.strip() for talker_id in self.talker_ids
        ):
            raise ValueError("talker_ids must not contain empty values")
        if self.clarification_question is not None and not self.clarification_question.strip():
            raise ValueError("clarification_question must not be empty")
        if self.talker_ids and self.clarification_question is not None:
            raise ValueError("speaker resolution cannot return identities and a clarification")
        object.__setattr__(
            self,
            "talker_ids",
            tuple(dict.fromkeys(talker_id.strip() for talker_id in self.talker_ids)),
        )
        if self.clarification_question is not None:
            object.__setattr__(
                self,
                "clarification_question",
                self.clarification_question.strip(),
            )


class TalkerNameResolver:
    """Resolve canonical names, first-last forms, and unique surnames from a cached catalog."""

    def __init__(self, catalog: TalkerCatalog) -> None:
        """Bind the lazy resolver to the Hansard speaker catalog."""
        self._catalog = catalog
        self._talkers: tuple[HansardSpeaker, ...] | None = None
        self._lock = Lock()

    def resolve(self, speaker_names: tuple[str, ...]) -> SpeakerResolution:
        """Return identities for each speaker name without guessing ambiguous matches."""
        if not speaker_names:
            return SpeakerResolution()
        talkers = self._load_catalog()
        resolved: list[str] = []
        for requested_name in speaker_names:
            normalized = _normalize_name(requested_name)
            exact = [
                talker
                for talker in talkers
                if talker.id is not None
                and normalized
                in {
                    _normalize_name(talker.name),
                    _normalize_name(f"{talker.first_name} {talker.last_name}"),
                }
            ]
            matches = exact
            if not matches:
                surname_matches = [
                    talker
                    for talker in talkers
                    if talker.id is not None and normalized == _normalize_name(talker.last_name)
                ]
                matches = surname_matches

            unique = {talker.id for talker in matches if talker.id is not None}
            if len(unique) != 1:
                if len(unique) > 1:
                    question = f"Which speaker named {requested_name.strip()} do you mean?"
                else:
                    question = (
                        f"I couldn't find a Hansard speaker named {requested_name.strip()}. "
                        "Could you clarify the speaker?"
                    )
                return SpeakerResolution(clarification_question=question)
            resolved.extend(unique)

        return SpeakerResolution(talker_ids=tuple(dict.fromkeys(resolved)))

    def _load_catalog(self) -> tuple[HansardSpeaker, ...]:
        """Load the catalog on first speaker lookup and retain it for the process lifetime."""
        if self._talkers is not None:
            return self._talkers
        with self._lock:
            if self._talkers is None:
                self._talkers = tuple(self._catalog.list_talkers())
        return self._talkers


def _normalize_name(name: str) -> str:
    """Normalize punctuation and case for exact speaker-name comparisons."""
    return " ".join(_NAME_TOKEN.sub(" ", name).casefold().split())
