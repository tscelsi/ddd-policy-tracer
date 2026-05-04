"""Operator CLI entrypoint and sitemap resolution helpers."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TextIO

from ..utils.logger import configure_logging, get_logger
from .service_layer import ingest_source_documents
from .source_strategies import get_source_strategy

LOGGER = get_logger(__name__, ctx="cli")


def run_cli(
    argv: Sequence[str],
    *,
    stdout: TextIO,
    fetch_document: Callable[..., tuple[str, bytes]] | None = None,
    fetch: Callable[[str, str], str] | None = None,
) -> int:
    """Execute operator CLI commands for manual acquisition runs."""
    _ = fetch_document
    if fetch is None:
        raise ValueError("fetch is required for source discovery")

    parser = argparse.ArgumentParser(
        prog="ddd-policy-tracer",
        description=(
            "Acquire source documents and persist normalized versions with "
            "audit metadata. Sources may use sitemap or listing discovery."
        ),
        epilog=(
            "Examples:\n"
            "  ddd-policy-tracer acquire --source australia_institute "
            "--state-path acquisition.db --artifact-dir artifacts\n"
            "  ddd-policy-tracer acquire --source lowy_institute "
            "--state-path acquisition.db --artifact-dir artifacts "
            "--published-within-years 2 --dry-run\n"
            "  ddd-policy-tracer acquire --source australia_institute "
            "--state-path acquisition.db --artifact-dir artifacts "
            "--published-within-years 2 --limit 50"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    acquire_parser = subparsers.add_parser(
        "acquire",
        help="Run one acquisition against a source",
        description=(
            "Discover source URLs and persist document versions. "
            "australia_institute uses sitemap discovery; lowy_institute uses "
            "paginated /publications?page=N listing discovery and requires "
            "--limit or --published-* guardrails."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    acquire_parser.add_argument(
        "--source",
        required=True,
        help=(
            "Source identifier (australia_institute or lowy_institute) "
            "stored on every persisted document version."
        ),
    )
    acquire_parser.add_argument(
        "--state-path",
        dest="state_path",
        required=True,
        help=(
            "Path to SQLite or JSONL file for persistence state. Should be "
            "SQLite when repository-backend=sqlite "
            "or JSONL when repository-backend=filesystem."
        ),
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
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of discovered URLs to process. For "
            "lowy_institute, this is also a required run guardrail unless "
            "--published-* is provided."
        ),
    )
    published_group = acquire_parser.add_mutually_exclusive_group()
    published_group.add_argument(
        "--published-since",
        default=None,
        help=(
            "Only process entries on/after this UTC timestamp (ISO-8601). "
            "For lowy_institute this bounds listing pagination.\n"
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
    acquire_parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="warning",
        help="Log verbosity level for acquisition diagnostics.",
    )

    args = parser.parse_args(list(argv))
    configure_logging(args.log_level)

    if args.command != "acquire":
        parser.print_help(file=stdout)
        return 2

    published_since = _resolve_published_since(
        published_since_raw=args.published_since,
        published_within_years=args.published_within_years,
    )
    cli_logger = LOGGER.bind(source=args.source, dry_run=args.dry_run)
    cli_logger.info(
        "cli acquire start backend=%s limit=%s published_since=%s",
        args.repository_backend,
        args.limit,
        _format_datetime(published_since),
    )

    _validate_source_constraints(
        source=args.source,
        limit=args.limit,
        published_since=published_since,
    )

    source_strategy = get_source_strategy(args.source)
    discovered_documents, discovery_scan_count = (
        source_strategy.discover_documents(
            fetch=fetch,
            published_since=published_since,
            limit=args.limit,
        )
    )
    discovered_urls = [entry.source_url for entry in discovered_documents]

    if args.dry_run:
        cli_logger.info(
            "%s dry-run discovery discovered=%s pages_scanned=%s",
            args.source,
            len(discovered_urls),
            discovery_scan_count,
        )
        dry_run_parts = [
            "dry_run",
            f"source={args.source}",
            f"discovered_urls={len(discovered_urls)}",
            f"pages_scanned={discovery_scan_count}",
            f"limit={args.limit}",
            f"published_since={_format_datetime(published_since)}",
        ]
        stdout.write(" ".join(dry_run_parts) + "\n")
        return 0

    cli_logger.info(
        "%s discovery completed discovered=%s pages_scanned=%s",
        args.source,
        len(discovered_documents),
        discovery_scan_count,
    )

    if discovered_documents is None or len(discovered_documents) == 0:
        cli_logger.warning("no discovered documents to process, exiting")
        stdout.write(
            f"source={args.source} processed_urls=0 ingested_documents=0 "
            f"failed_documents=0 skipped_urls=0 run_status=no_discoveries\n",
        )
        return 0

    report = ingest_source_documents(
        source_id=args.source,
        artifact_dir=Path(args.artifact_dir),
        discovered_documents=discovered_documents,
        state_path=Path(args.state_path),
        repository_backend=args.repository_backend,
    )

    output_parts = [
        f"source={args.source}",
        f"processed_urls={report.processed_urls}",
        f"ingested_documents={report.ingested_documents}",
        f"failed_documents={report.failed_documents}",
        f"skipped_urls={report.skipped_urls}",
        f"run_status={report.run_status}",
    ]
    if report.skipped_reasons:
        output_parts.extend(
            [
                f"skip_reasons={len(report.skipped_reasons)}",
                f"skip_reason={report.skipped_reasons[0]}",
            ],
        )
        for reason in report.skipped_reasons:
            cli_logger.info("skip reason: %s", reason)
    if report.document_failures:
        for failure in report.document_failures:
            cli_logger.error("document failure: %s", failure)
    cli_logger.info(
        "cli acquire complete processed=%s ingested=%s "
        "skipped=%s failed=%s status=%s",
        report.processed_urls,
        report.ingested_documents,
        report.skipped_urls,
        report.failed_documents,
        report.run_status,
    )
    stdout.write(" ".join(output_parts) + "\n")
    return 0


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
                "for example 2024-01-01T00:00:00+00:00",
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


def _validate_source_constraints(
    *,
    source: str,
    limit: int | None,
    published_since: datetime | None,
) -> None:
    """Validate source-specific CLI constraints before execution."""
    if source == "lowy_institute":
        if limit is None and published_since is None:
            raise ValueError(
                "lowy_institute requires either --limit or --published-*",
            )
        return
