"""Operator CLI entrypoint and sitemap resolution helpers."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TextIO

from .service_layer import ingest_source_documents
from .source_strategies import LowyInstituteSourceStrategy, get_source_strategy


def run_cli(
    argv: Sequence[str],
    *,
    fetch_document: Callable[..., tuple[str, bytes]],
    stdout: TextIO,
    fetch_text_url: Callable[[str, str], str] | None = None,
) -> int:
    """Execute operator CLI commands for manual acquisition runs."""
    parser = argparse.ArgumentParser(
        prog="ddd-policy-tracer",
        description=(
            "Acquire source documents from sitemaps and persist normalized "
            "versions with audit metadata."
        ),
        epilog=(
            "Examples:\n"
            "  ddd-policy-tracer acquire --source australia_institute "
            "--sitemap-url https://australiainstitute.org.au/sitemap_index.xml "
            "--child-sitemap-pattern tai_cpt_report-sitemap "
            "--sqlite-path acquisition.db --artifact-dir artifacts\n"
            "  ddd-policy-tracer acquire --source australia_institute "
            "--sitemap-xml-path sitemap.xml --sqlite-path acquisition.db "
            "--artifact-dir artifacts --published-within-years 2 --limit 50"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    acquire_parser = subparsers.add_parser(
        "acquire",
        help="Run one acquisition against a source",
        description=(
            "Discover report URLs from a sitemap, optionally filter by "
            "publication time, then fetch and persist document versions."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    acquire_parser.add_argument(
        "--source",
        required=True,
        help="Source identifier stored on every persisted document version.",
    )
    sitemap_group = acquire_parser.add_mutually_exclusive_group(required=False)
    sitemap_group.add_argument(
        "--sitemap-xml-path",
        help="Path to a local sitemap XML file (urlset or sitemapindex).",
    )
    sitemap_group.add_argument(
        "--sitemap-url",
        help="Remote sitemap URL to fetch over HTTP (supports sitemapindex).",
    )
    acquire_parser.add_argument(
        "--child-sitemap-pattern",
        default=None,
        help=(
            "Substring filter applied to child sitemap URLs when --sitemap-url "
            "points to a sitemap index."
        ),
    )
    acquire_parser.add_argument(
        "--sqlite-path",
        required=True,
        help="Path to SQLite database file or JSONL state file.",
    )
    acquire_parser.add_argument(
        "--repository-backend",
        choices=["sqlite", "filesystem"],
        default="sqlite",
        help=(
            "Persistence adapter. Use 'sqlite' for relational storage or "
            "'filesystem' for JSONL state."
        ),
    )
    acquire_parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Directory where raw fetched artifacts are written.",
    )
    acquire_parser.add_argument(
        "--user-agent",
        default="ddd-policy-tracer/0.1",
        help="User-Agent header sent on HTTP requests.",
    )
    acquire_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of discovered sitemap URLs to process.",
    )
    published_group = acquire_parser.add_mutually_exclusive_group()
    published_group.add_argument(
        "--published-since",
        default=None,
        help=(
            "Only process entries whose sitemap lastmod is on/after this UTC "
            "timestamp (ISO-8601).\n"
            "Example: 2024-01-01T00:00:00+00:00"
        ),
    )
    published_group.add_argument(
        "--published-within-years",
        type=float,
        default=None,
        help=(
            "Only process entries newer than now minus this many years.\n"
            "Example: --published-within-years 2"
        ),
    )
    acquire_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without fetching or persisting.",
    )

    args = parser.parse_args(list(argv))

    if args.command != "acquire":
        parser.print_help(file=stdout)
        return 2

    published_since = _resolve_published_since(
        published_since_raw=args.published_since,
        published_within_years=args.published_within_years,
    )

    _validate_source_constraints(
        source=args.source,
        sitemap_xml_path=args.sitemap_xml_path,
        sitemap_url=args.sitemap_url,
        limit=args.limit,
        published_since=published_since,
    )

    sitemap_xml = ""
    selected_sitemaps = 0
    if args.source != "lowy_institute":
        sitemap_xml, selected_sitemaps = _load_sitemap_xml(
            sitemap_xml_path=args.sitemap_xml_path,
            sitemap_url=args.sitemap_url,
            child_sitemap_pattern=args.child_sitemap_pattern,
            user_agent=args.user_agent,
            fetch_text_url=fetch_text_url,
        )

    if args.dry_run:
        if args.source == "lowy_institute":
            strategy = get_source_strategy(args.source)
            if not isinstance(strategy, LowyInstituteSourceStrategy):
                raise ValueError("lowy_institute strategy is not available")
            if fetch_text_url is None:
                raise ValueError(
                    "fetch_text_url is required for lowy_institute discovery"
                )
            discovered_entries, selected_sitemaps = (
                strategy.discover_listing_entries(
                    fetch_text_url=fetch_text_url,
                    user_agent=args.user_agent,
                    published_since=published_since,
                    limit=args.limit,
                )
            )
            discovered = [entry.source_url for entry in discovered_entries]
        else:
            discovered = _discover_urls_with_limit(
                sitemap_xml=sitemap_xml,
                limit=args.limit,
                published_since=published_since,
            )
        stdout.write(
            " ".join(
                [
                    "dry_run",
                    f"source={args.source}",
                    f"discovered_urls={len(discovered)}",
                    f"selected_sitemaps={selected_sitemaps}",
                    f"limit={args.limit}",
                    f"published_since={_format_datetime(published_since)}",
                ]
            )
            + "\n"
        )
        return 0

    pre_discovered_entries = None
    if args.source == "lowy_institute":
        strategy = get_source_strategy(args.source)
        if not isinstance(strategy, LowyInstituteSourceStrategy):
            raise ValueError("lowy_institute strategy is not available")
        if fetch_text_url is None:
            raise ValueError(
                "fetch_text_url is required for lowy_institute discovery"
            )
        pre_discovered_entries, selected_sitemaps = (
            strategy.discover_listing_entries(
                fetch_text_url=fetch_text_url,
                user_agent=args.user_agent,
                published_since=published_since,
                limit=args.limit,
            )
        )

    report = ingest_source_documents(
        source_id=args.source,
        sitemap_xml=sitemap_xml,
        sqlite_path=Path(args.sqlite_path),
        artifact_dir=Path(args.artifact_dir),
        fetch_document=fetch_document,
        user_agent=args.user_agent,
        repository_backend=args.repository_backend,
        limit=args.limit,
        published_since=published_since,
        discovered_entries=pre_discovered_entries,
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
    *,
    sitemap_xml: str,
    limit: int | None,
    published_since: datetime | None = None,
) -> list[str]:
    """Discover URLs from sitemap XML and apply an optional upper bound."""
    from .adapters import discover_sitemap_entries

    urls = [
        entry.source_url
        for entry in discover_sitemap_entries(sitemap_xml)
        if _is_entry_published_on_or_after(
            entry_published_at=entry.published_at,
            published_since=published_since,
        )
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


def _resolve_published_since(
    *,
    published_since_raw: str | None,
    published_within_years: float | None,
) -> datetime | None:
    """Resolve publish-time filtering options into one UTC timestamp."""
    if published_since_raw is not None:
        parsed = _parse_datetime(published_since_raw)
        if parsed is None:
            raise ValueError(
                "--published-since must be ISO-8601, "
                "for example 2024-01-01T00:00:00+00:00"
            )
        return parsed

    if published_within_years is None:
        return None
    if published_within_years <= 0:
        raise ValueError("--published-within-years must be greater than 0")

    days = published_within_years * 365.25
    return datetime.now(UTC) - timedelta(days=days)


def _parse_datetime(value: str) -> datetime | None:
    """Parse one datetime string into a UTC-aware timestamp."""
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_datetime(value: datetime | None) -> str:
    """Format one optional datetime for CLI output."""
    if value is None:
        return "none"
    return value.isoformat()


def _is_entry_published_on_or_after(
    *, entry_published_at: str | None, published_since: datetime | None
) -> bool:
    """Return true when an entry passes the optional publish-time filter."""
    if published_since is None:
        return True
    if entry_published_at is None:
        return False

    entry_timestamp = _parse_datetime(entry_published_at)
    if entry_timestamp is None:
        return False
    return entry_timestamp >= published_since


def _validate_source_constraints(
    *,
    source: str,
    sitemap_xml_path: str | None,
    sitemap_url: str | None,
    limit: int | None,
    published_since: datetime | None,
) -> None:
    """Validate source-specific CLI constraints before execution."""
    has_sitemap_input = sitemap_xml_path is not None or sitemap_url is not None

    if source == "lowy_institute":
        if limit is None and published_since is None:
            raise ValueError(
                "lowy_institute requires either --limit or --published-*"
            )
        return

    if not has_sitemap_input:
        raise ValueError(
            "Either --sitemap-xml-path or --sitemap-url is required"
        )
