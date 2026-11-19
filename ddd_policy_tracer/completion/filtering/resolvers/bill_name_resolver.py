"""Module for resolving a bill identifier against the Hansard bill catalog."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


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
