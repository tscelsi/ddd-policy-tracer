"""Top-level CLI that routes to discovery and analysis commands."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TextIO

from .analysis.cli import run_cli as run_analysis_cli
from .discovery.cli import run_cli as run_discovery_cli


def run_cli(
    argv: Sequence[str],
    *,
    stdout: TextIO,
    fetch: Callable[[str, str], str] | None = None,
) -> int:
    """Route CLI requests to discovery acquire or analysis chunk flows."""
    args = list(argv)
    if not args:
        return run_discovery_cli(args, stdout=stdout, fetch=fetch)

    command = args[0]
    if command == "acquire":
        return run_discovery_cli(args, stdout=stdout, fetch=fetch)
    if command == "chunk":
        return run_analysis_cli(args, stdout=stdout)

    return run_discovery_cli(args, stdout=stdout, fetch=fetch)
