"""Module entrypoint for top-level CLI commands."""

from __future__ import annotations

import sys

from .cli import run_cli


def main() -> int:
    """Run top-level CLI using module invocation arguments."""
    return run_cli(sys.argv[1:], stdout=sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
