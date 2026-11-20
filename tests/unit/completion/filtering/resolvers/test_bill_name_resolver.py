"""Tests for resolving English bill names against the Hansard bill catalog."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from ddd_policy_tracer.completion.filtering.resolvers.bill_name_resolver import (
    BillResolution,
    MongoBillNameResolver,
)
from ddd_policy_tracer.ingestion.hansard.bill_repository import Bill

_UNIVERSITIES_BILL = Bill(
    id="r7311",
    name=(
        "Universities Accord (National Higher Education Code to Prevent and Respond "
        "to Gender-based Violence) (Consequential Amendments) Bill 2025"
    ),
)


class FakeBillRepository:
    """Return a fixed bill catalog while counting catalog reads."""

    def __init__(self, bills: Sequence[Bill]) -> None:
        """Store the catalog and initialize the read counter."""
        self.bills = list(bills)
        self.read_count = 0

    async def list_bills(self) -> list[Bill]:
        """Return the configured bill catalog."""
        self.read_count += 1
        return self.bills

    async def save_bills(self, bills: list[Bill]) -> None:
        """Replace the configured catalog for contract compatibility."""
        self.bills = bills

    async def get_bill_by_id(self, id: str) -> Bill | None:
        """Return the bill with the requested identifier."""
        return next((bill for bill in self.bills if bill.id == id), None)


@pytest.mark.parametrize(
    "query",
    [
        _UNIVERSITIES_BILL.name,
        "Universities Accord",
        "Uni Accord",
        "Universties Accord",
        "gender based violence consequential amendments",
    ],
)
async def test_bill_name_matches_full_shortened_and_colloquial_names(query: str) -> None:
    """Resolve canonical titles and meaningful shortened or abbreviated forms."""
    resolver = MongoBillNameResolver(FakeBillRepository([_UNIVERSITIES_BILL]))

    result = await resolver.resolve([query])

    assert result == BillResolution(bill_ids=("r7311",))


async def test_resolver_matches_canonical_id_case_insensitively() -> None:
    """Resolve an explicit bill identifier as an exact match."""
    resolver = MongoBillNameResolver(FakeBillRepository([_UNIVERSITIES_BILL]))

    result = await resolver.resolve(["R7311"])

    assert result == BillResolution(bill_ids=("r7311",))


@pytest.mark.parametrize("query", ["", " ", "education bill"])
async def test_bill_name_rejects_blank_or_generic_names(query: str) -> None:
    """Ask for clarification instead of resolving blank or generic names."""
    resolver = MongoBillNameResolver(FakeBillRepository([_UNIVERSITIES_BILL]))

    result = await resolver.resolve([query])

    assert result.bill_ids == ()
    assert result.clarification_question is not None


async def test_resolver_returns_bill_id_for_short_name_and_caches_catalog() -> None:
    """Resolve a shortened name and reuse the loaded bill catalog."""
    repository = FakeBillRepository([_UNIVERSITIES_BILL])
    resolver = MongoBillNameResolver(repository)

    first = await resolver.resolve(["Universities Accord"])
    second = await resolver.resolve(["Uni Accord"])

    assert first == BillResolution(bill_ids=("r7311",))
    assert second == BillResolution(bill_ids=("r7311",))
    assert repository.read_count == 1


async def test_resolver_requests_clarification_for_ambiguous_short_name() -> None:
    """Do not choose arbitrarily when a shortened title matches multiple bills."""
    repository = FakeBillRepository(
        [
            _UNIVERSITIES_BILL,
            Bill(
                id="r9000",
                name="Universities Accord (Implementation) Bill 2025",
            ),
        ],
    )
    resolver = MongoBillNameResolver(repository)

    result = await resolver.resolve(["Universities Accord"])

    assert result.bill_ids == ()
    assert result.clarification_question is not None
    assert "multiple matches" in result.clarification_question


async def test_resolver_returns_empty_result_without_loading_catalog_for_no_names() -> None:
    """Avoid a repository read when there is no bill constraint to resolve."""
    repository = FakeBillRepository([_UNIVERSITIES_BILL])

    result = await MongoBillNameResolver(repository).resolve([])

    assert result == BillResolution()
    assert repository.read_count == 0
