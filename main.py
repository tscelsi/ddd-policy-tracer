from __future__ import annotations

import sys
from typing import Sequence, cast
from urllib.request import Request, urlopen

from ddd_policy_tracer.cli import run_cli


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    return run_cli(
        args,
        fetch_document=fetch_document_over_http,
        fetch_text_url=fetch_text_url,
        stdout=sys.stdout,
    )


def fetch_document_over_http(url: str, user_agent: str) -> tuple[str, bytes]:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=30) as response:
        content_type = response.headers.get_content_type() or "application/octet-stream"
        payload = response.read()
    return content_type, payload


def fetch_text_url(url: str, user_agent: str) -> str:
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=30) as response:
        payload = cast(bytes, response.read())
        return payload.decode("utf-8", errors="ignore")


if __name__ == "__main__":
    raise SystemExit(main())
