from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence, TextIO

from .service_layer import ingest_source_documents


def run_cli(
    argv: Sequence[str],
    *,
    fetch_document: Callable[..., tuple[str, bytes]],
    stdout: TextIO,
) -> int:
    parser = argparse.ArgumentParser(prog="ddd-policy-tracer")
    subparsers = parser.add_subparsers(dest="command")

    acquire_parser = subparsers.add_parser("acquire")
    acquire_parser.add_argument("--source", required=True)
    acquire_parser.add_argument("--sitemap-xml-path", required=True)
    acquire_parser.add_argument("--sqlite-path", required=True)
    acquire_parser.add_argument("--artifact-dir", required=True)
    acquire_parser.add_argument("--limit", type=int, default=None)
    acquire_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(list(argv))

    if args.command != "acquire":
        parser.print_help(file=stdout)
        return 2

    sitemap_xml = Path(args.sitemap_xml_path).read_text(encoding="utf-8")

    if args.dry_run:
        discovered = _discover_urls_with_limit(sitemap_xml=sitemap_xml, limit=args.limit)
        stdout.write(
            f"dry_run source={args.source} discovered_urls={len(discovered)} limit={args.limit}\n"
        )
        return 0

    report = ingest_source_documents(
        source_id=args.source,
        sitemap_xml=sitemap_xml,
        sqlite_path=Path(args.sqlite_path),
        artifact_dir=Path(args.artifact_dir),
        fetch_document=fetch_document,
        limit=args.limit,
    )

    stdout.write(
        " ".join(
            [
                f"source={args.source}",
                f"processed_urls={report.processed_urls}",
                f"ingested_documents={report.ingested_documents}",
                f"failed_documents={report.failed_documents}",
                f"skipped_urls={report.skipped_urls}",
                f"run_status={report.run_status}",
            ]
        )
        + "\n"
    )
    return 0


def _discover_urls_with_limit(*, sitemap_xml: str, limit: int | None) -> list[str]:
    from .adapters import discover_urls_from_sitemap

    urls = discover_urls_from_sitemap(sitemap_xml)
    if limit is not None:
        return urls[: max(0, limit)]
    return urls
