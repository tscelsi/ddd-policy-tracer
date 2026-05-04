"""Runtime bootstrap for the operator acquisition CLI."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import cast
from urllib.request import Request, urlopen

from ddd_policy_tracer.cli import run_cli


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI entrypoint and return a process exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    return run_cli(
        args,
        fetch=fetch_text_url,
        stdout=sys.stdout,
    )


def fetch_text_url(url: str, user_agent: str) -> str:
    """Fetch a text URL payload, used for sitemap document retrieval."""
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=30) as response:
        payload = cast(bytes, response.read())
        return payload.decode("utf-8", errors="ignore")


if __name__ == "__main__":
    raise SystemExit(main())
