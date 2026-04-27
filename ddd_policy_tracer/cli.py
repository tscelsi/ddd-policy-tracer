"""Operator CLI entrypoint and sitemap resolution helpers."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from .service_layer import ingest_source_documents


def run_cli(
    argv: Sequence[str],
    *,
    fetch_document: Callable[..., tuple[str, bytes]],
    stdout: TextIO,
    fetch_text_url: Callable[[str, str], str] | None = None,
) -> int:
    """Execute operator CLI commands for manual acquisition runs."""
    parser = argparse.ArgumentParser(prog="ddd-policy-tracer")
    subparsers = parser.add_subparsers(dest="command")

    acquire_parser = subparsers.add_parser("acquire")
    acquire_parser.add_argument("--source", required=True)
    sitemap_group = acquire_parser.add_mutually_exclusive_group(required=True)
    sitemap_group.add_argument("--sitemap-xml-path")
    sitemap_group.add_argument("--sitemap-url")
    acquire_parser.add_argument("--child-sitemap-pattern", default=None)
    acquire_parser.add_argument("--sqlite-path", required=True)
    acquire_parser.add_argument(
        "--repository-backend",
        choices=["sqlite", "filesystem"],
        default="sqlite",
    )
    acquire_parser.add_argument("--artifact-dir", required=True)
    acquire_parser.add_argument("--user-agent", default="ddd-policy-tracer/0.1")
    acquire_parser.add_argument("--limit", type=int, default=None)
    acquire_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(list(argv))

    if args.command != "acquire":
        parser.print_help(file=stdout)
        return 2

    sitemap_xml, selected_sitemaps = _load_sitemap_xml(
        sitemap_xml_path=args.sitemap_xml_path,
        sitemap_url=args.sitemap_url,
        child_sitemap_pattern=args.child_sitemap_pattern,
        user_agent=args.user_agent,
        fetch_text_url=fetch_text_url,
    )

    if args.dry_run:
        discovered = _discover_urls_with_limit(
            sitemap_xml=sitemap_xml, limit=args.limit
        )
        stdout.write(
            " ".join(
                [
                    "dry_run",
                    f"source={args.source}",
                    f"discovered_urls={len(discovered)}",
                    f"selected_sitemaps={selected_sitemaps}",
                    f"limit={args.limit}",
                ]
            )
            + "\n"
        )
        return 0

    report = ingest_source_documents(
        source_id=args.source,
        sitemap_xml=sitemap_xml,
        sqlite_path=Path(args.sqlite_path),
        artifact_dir=Path(args.artifact_dir),
        fetch_document=fetch_document,
        user_agent=args.user_agent,
        repository_backend=args.repository_backend,
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


def _discover_urls_with_limit(
    *, sitemap_xml: str, limit: int | None
) -> list[str]:
    """Discover URLs from sitemap XML and apply an optional upper bound."""
    from .adapters import discover_sitemap_entries

    urls = [
        entry.source_url
        for entry in discover_sitemap_entries(sitemap_xml)
    ]
    if limit is not None:
        return urls[: max(0, limit)]
    return urls


def _load_sitemap_xml(
    *,
    sitemap_xml_path: str | None,
    sitemap_url: str | None,
    child_sitemap_pattern: str | None,
    user_agent: str,
    fetch_text_url: Callable[[str, str], str] | None,
) -> tuple[str, int]:
    """Load sitemap XML from local file or URL, resolving indexes as needed."""
    if sitemap_xml_path is not None:
        return Path(sitemap_xml_path).read_text(encoding="utf-8"), 1

    if sitemap_url is None:
        raise ValueError(
            "Either --sitemap-xml-path or --sitemap-url is required"
        )

    if fetch_text_url is None:
        raise ValueError("fetch_text_url is required when using --sitemap-url")

    root_xml = fetch_text_url(sitemap_url, user_agent)
    if _is_sitemap_index(root_xml):
        child_sitemaps = _discover_child_sitemaps(root_xml)
        if child_sitemap_pattern is not None:
            child_sitemaps = [
                url for url in child_sitemaps if child_sitemap_pattern in url
            ]
        child_urlsets = [
            fetch_text_url(url, user_agent) for url in child_sitemaps
        ]
        return _merge_urlsets(child_urlsets), len(child_sitemaps)

    return root_xml, 1


def _is_sitemap_index(xml_text: str) -> bool:
    """Return true when the XML payload is a sitemap index document."""
    root = ET.fromstring(xml_text)
    return root.tag.endswith("sitemapindex")


def _discover_child_sitemaps(sitemap_index_xml: str) -> list[str]:
    """Extract child sitemap URLs from a sitemap index document."""
    root = ET.fromstring(sitemap_index_xml)
    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [
        node.text.strip()
        for node in root.findall("sm:sitemap/sm:loc", namespace)
        if node.text
    ]


def _merge_urlsets(urlset_xml_documents: list[str]) -> str:
    """Merge URL set documents into one deduplicated URL set XML payload."""
    from .adapters import discover_sitemap_entries

    merged_entries: dict[str, str | None] = {}
    for xml_text in urlset_xml_documents:
        for entry in discover_sitemap_entries(xml_text):
            existing = merged_entries.get(entry.source_url)
            if existing is None:
                merged_entries[entry.source_url] = entry.published_at
                continue

            if entry.published_at is None:
                continue
            if existing is None or entry.published_at > existing:
                merged_entries[entry.source_url] = entry.published_at

    lines = [
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for url, published_at in merged_entries.items():
        if published_at is None:
            lines.append(f"  <url><loc>{url}</loc></url>")
        else:
            lines.append(
                "  <url>"
                f"<loc>{url}</loc>"
                f"<lastmod>{published_at}</lastmod>"
                "</url>"
            )
    lines.extend([
        "</urlset>",
    ])
    return "\n".join(lines)
