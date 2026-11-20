"""Module for resolving a bill identifier against the Hansard bill catalog."""

from __future__ import annotations

import asyncio
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol

from ddd_policy_tracer.ingestion.hansard.bill_repository import AbstractBillRepository, Bill

_WORD = re.compile(r"[^\W_]+", flags=re.UNICODE)
_NON_INFORMATIVE_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "bill",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "this",
    "that",
    "to",
    "with",
    "without",
}
_NON_PLURAL_WORDS = {"analysis", "news", "politics", "series", "species", "status"}


@dataclass(frozen=True)
class BillResolution:
    """Return resolved bill IDs or a question for an ambiguous bill name."""

    bill_ids: tuple[str, ...] = ()
    clarification_question: str | None = None


class AbstractBillNameResolver(Protocol):
    """Port for resolving a bill identifier against the Hansard bill catalog."""

    async def resolve(self, bill_names: Sequence[str]) -> BillResolution:
        """Return resolved bill IDs or a question for an ambiguous bill name."""
        ...


class MongoBillNameResolver(AbstractBillNameResolver):
    """Resolve canonical and shortened bill names against a cached catalog."""

    def __init__(self, repository: AbstractBillRepository) -> None:
        """Accept the Mongo-backed bill repository port."""
        self._repository = repository
        self._bills: tuple[Bill, ...] | None = None
        self._catalog_lock = asyncio.Lock()

    async def resolve(self, bill_names: Sequence[str]) -> BillResolution:
        """Return unique bill IDs, or a clarification for the first unresolved name."""
        if not bill_names:
            return BillResolution()
        bills = await self._load_catalog()
        resolved_ids: list[str] = []
        for bill_name in bill_names:
            matches = {bill.id for bill in bills if _matches_bill(bill_name, bill)}
            if len(matches) != 1:
                reason = "I couldn't find" if not matches else "I found multiple matches for"
                return BillResolution(
                    clarification_question=(
                        f"{reason} the Hansard bill “{bill_name}”. "
                        "Please provide a more specific identifier, such as the bill's "
                        "full name or a unique ID."
                    ),
                )
            resolved_ids.extend(matches)
        return BillResolution(bill_ids=tuple(dict.fromkeys(resolved_ids)))

    async def _load_catalog(self) -> tuple[Bill, ...]:
        """Load the bill catalog once, preserving a single concurrent repository read."""
        if self._bills is not None:
            return self._bills
        async with self._catalog_lock:
            if self._bills is None:
                self._bills = tuple(await self._repository.list_bills())
        return self._bills


def _matches_bill(bill_name: str, bill: Bill) -> bool:
    """Match exact IDs, shortened names, common abbreviations, and minor misspellings."""
    all_query_tokens = _tokens(bill_name)
    if not all_query_tokens:
        return False

    if all_query_tokens == _tokens(bill.id) or all_query_tokens == _tokens(bill.name):
        return True

    query_tokens = tuple(token for token in all_query_tokens if _is_informative(token))
    bill_tokens = tuple(token for token in _tokens(bill.name) if _is_informative(token))
    if len(query_tokens) < 2:
        return False
    if not bill_tokens:
        return False

    candidates = [
        [
            index
            for index, bill_token in enumerate(bill_tokens)
            if _tokens_match(query_token, bill_token)
        ]
        for query_token in query_tokens
    ]
    if any(not matching_indices for matching_indices in candidates):
        return False

    return _can_match_distinct_tokens(candidates)


def _can_match_distinct_tokens(candidates: list[list[int]]) -> bool:
    """Check that each query word maps to a different word in the bill title."""
    ordered_queries = sorted(range(len(candidates)), key=lambda index: len(candidates[index]))
    matched_queries: dict[int, int] = {}

    def assign(query_index: int, visited_titles: set[int]) -> bool:
        for title_index in candidates[query_index]:
            if title_index in visited_titles:
                continue
            visited_titles.add(title_index)
            previous_query = matched_queries.get(title_index)
            if previous_query is None or assign(previous_query, visited_titles):
                matched_queries[title_index] = query_index
                return True
        return False

    return all(assign(query_index, set()) for query_index in ordered_queries)


def _tokens_match(query_token: str, bill_token: str) -> bool:
    """Match normalized words, useful prefixes, or closely spelled variants."""
    if query_token == bill_token:
        return True
    if len(query_token) >= 3 and bill_token.startswith(query_token):
        return True
    return (
        min(len(query_token), len(bill_token)) >= 5
        and SequenceMatcher(None, query_token, bill_token).ratio() >= 0.84
    )


def _tokens(value: str) -> tuple[str, ...]:
    """Normalize Unicode bill text into case-folded, punctuation-free word tokens."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(_normalize_plural(token) for token in _WORD.findall(normalized))


def _normalize_plural(token: str) -> str:
    """Normalize common English plural endings without stemming other words."""
    if token in _NON_PLURAL_WORDS:
        return token
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("is", "ss", "us")):
        return token[:-1]
    return token


def _is_informative(token: str) -> bool:
    """Ignore legal boilerplate and years when assessing a shortened bill title."""
    return token not in _NON_INFORMATIVE_WORDS and not token.isdigit()
