"""Application service orchestration for document acquisition use cases."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from .adapters import (
    DiskArtifactStore,
    FilesystemSourceDocumentRepository,
    SQLiteSourceDocumentRepository,
    discover_sitemap_entries,
    extract_pdf_urls_from_report_html,
    extract_text_from_pdf_bytes,
)
from .domain import (
    SourceDocumentVersion,
    compute_checksum,
    normalize_source_document_id,
    normalize_text,
)


@dataclass(frozen=True)
class AcquisitionReport:
    """Capture aggregate run results and emitted acquisition events."""

    run_id: str
    processed_urls: int
    ingested_documents: int
    failed_documents: int
    skipped_urls: int
    retry_attempts: int
    document_failures: tuple[str, ...]
    events: tuple[AcquisitionEvent, ...]
    run_status: Literal["completed", "completed_with_failures", "failed"]


@dataclass(frozen=True)
class AcquisitionEvent:
    """Represent one domain event emitted during an acquisition run."""

    event_type: Literal[
        "AcquisitionRunStarted",
        "SourceDocumentIngested",
        "SourceDocumentIngestionFailed",
        "AcquisitionRunCompleted",
    ]
    run_id: str
    source_id: str
    source_url: str | None = None
    source_document_id: str | None = None
    run_status: (
        Literal["completed", "completed_with_failures", "failed"] | None
    ) = None


def ingest_source_documents(
    *,
    source_id: str,
    sitemap_xml: str,
    sqlite_path: Path,
    artifact_dir: Path,
    fetch_document: Callable[..., tuple[str, bytes]],
    user_agent: str = "ddd-policy-tracer/0.1",
    repository_backend: Literal["sqlite", "filesystem"] = "sqlite",
    is_allowed_by_robots: Callable[[str, str], bool] | None = None,
    max_retries: int = 2,
    backoff_seconds: Sequence[float] = (0.25, 0.5),
    sleep_fn: Callable[[float], None] = time.sleep,
    limit: int | None = None,
) -> AcquisitionReport:
    """Ingest discovered URLs and return aggregate acquisition outcomes."""
    repository = _build_repository(
        sqlite_path=sqlite_path,
        repository_backend=repository_backend,
    )
    artifact_store = DiskArtifactStore(artifact_dir)

    processed_urls = 0
    ingested_documents = 0
    failed_documents = 0
    skipped_urls = 0
    retry_attempts = 0
    document_failures: list[str] = []
    run_id = str(uuid4())
    events: list[AcquisitionEvent] = [
        AcquisitionEvent(
            event_type="AcquisitionRunStarted",
            run_id=run_id,
            source_id=source_id,
        )
    ]

    robots_checker = is_allowed_by_robots or (lambda _url, _ua: True)

    discovered_entries = discover_sitemap_entries(sitemap_xml)
    if limit is not None:
        discovered_entries = discovered_entries[: max(0, limit)]

    for entry in discovered_entries:
        source_url = entry.source_url
        processed_urls += 1

        if not robots_checker(source_url, user_agent):
            skipped_urls += 1
            continue

        source_document_id = normalize_source_document_id(source_url)

        try:
            content_type, raw_content, used_retries = _fetch_with_retries(
                fetch_document=fetch_document,
                source_url=source_url,
                user_agent=user_agent,
                max_retries=max_retries,
                backoff_seconds=backoff_seconds,
                sleep_fn=sleep_fn,
            )
            retry_attempts += used_retries
            if content_type != "text/html":
                raise ValueError("report page must be HTML")

            pdf_url = _select_pdf_url_from_report_html(
                report_url=source_url,
                report_html=raw_content,
            )
            (
                pdf_content_type,
                pdf_content,
                pdf_retries,
            ) = _fetch_with_retries(
                fetch_document=fetch_document,
                source_url=pdf_url,
                user_agent=user_agent,
                max_retries=max_retries,
                backoff_seconds=backoff_seconds,
                sleep_fn=sleep_fn,
            )
            retry_attempts += pdf_retries
            if pdf_content_type != "application/pdf":
                raise ValueError("selected report file is not a PDF")

            now = _utc_now_isoformat()
            extracted_text = extract_text_from_pdf_bytes(pdf_content)
            normalized = normalize_text(extracted_text)
            if not normalized:
                raise ValueError("normalized_text is empty")

            checksum = compute_checksum(pdf_content)
            latest_version = repository.get_latest(
                source_id=source_id,
                source_document_id=source_document_id,
            )
            if (
                latest_version is not None
                and latest_version.checksum == checksum
            ):
                continue

            raw_content_ref = artifact_store.store(
                source_document_id=source_document_id,
                content=pdf_content,
            )
            version = SourceDocumentVersion(
                source_id=source_id,
                source_document_id=source_document_id,
                source_url=source_url,
                published_at=entry.published_at,
                retrieved_at=now,
                checksum=checksum,
                normalized_text=normalized,
                raw_content_ref=raw_content_ref,
                content_type=pdf_content_type,
                created_at=now,
                updated_at=now,
            )
            repository.add_version(version)
            ingested_documents += 1
            events.append(
                AcquisitionEvent(
                    event_type="SourceDocumentIngested",
                    run_id=run_id,
                    source_id=source_id,
                    source_url=source_url,
                    source_document_id=source_document_id,
                )
            )
        except Exception as exc:
            failed_documents += 1
            document_failures.append(f"{source_url}: {exc}")
            events.append(
                AcquisitionEvent(
                    event_type="SourceDocumentIngestionFailed",
                    run_id=run_id,
                    source_id=source_id,
                    source_url=source_url,
                    source_document_id=source_document_id,
                )
            )

    if failed_documents == 0:
        run_status: Literal[
            "completed", "completed_with_failures", "failed"
        ] = "completed"
    elif ingested_documents == 0:
        run_status = "failed"
    else:
        run_status = "completed_with_failures"

    events.append(
        AcquisitionEvent(
            event_type="AcquisitionRunCompleted",
            run_id=run_id,
            source_id=source_id,
            run_status=run_status,
        )
    )

    return AcquisitionReport(
        run_id=run_id,
        processed_urls=processed_urls,
        ingested_documents=ingested_documents,
        failed_documents=failed_documents,
        skipped_urls=skipped_urls,
        retry_attempts=retry_attempts,
        document_failures=tuple(document_failures),
        events=tuple(events),
        run_status=run_status,
    )


def get_source_document_versions(
    *,
    sqlite_path: Path,
    source_id: str,
    repository_backend: Literal["sqlite", "filesystem"] = "sqlite",
) -> list[SourceDocumentVersion]:
    """Load persisted source document versions for one source."""
    repository = _build_repository(
        sqlite_path=sqlite_path,
        repository_backend=repository_backend,
    )
    return repository.list_versions(source_id=source_id)


def _call_fetch_document(
    *,
    fetch_document: Callable[..., tuple[str, bytes]],
    source_url: str,
    user_agent: str,
) -> tuple[str, bytes]:
    """Call the fetcher using supported URL-only or URL+agent signatures."""
    signature = inspect.signature(fetch_document)
    if len(signature.parameters) >= 2:
        return fetch_document(source_url, user_agent)
    return fetch_document(source_url)


def _fetch_with_retries(
    *,
    fetch_document: Callable[..., tuple[str, bytes]],
    source_url: str,
    user_agent: str,
    max_retries: int,
    backoff_seconds: Sequence[float],
    sleep_fn: Callable[[float], None],
) -> tuple[str, bytes, int]:
    """Fetch one source URL with bounded retries for transient failures."""
    retries_used = 0

    while True:
        try:
            content_type, raw_content = _call_fetch_document(
                fetch_document=fetch_document,
                source_url=source_url,
                user_agent=user_agent,
            )
            return content_type, raw_content, retries_used
        except (TimeoutError, ConnectionError):
            if retries_used >= max_retries:
                raise

            delay_index = min(retries_used, max(0, len(backoff_seconds) - 1))
            delay = backoff_seconds[delay_index] if backoff_seconds else 0.0
            sleep_fn(delay)
            retries_used += 1


def _select_pdf_url_from_report_html(
    *, report_url: str, report_html: bytes
) -> str:
    """Choose one PDF URL from a report page or raise when none exist."""
    pdf_urls = extract_pdf_urls_from_report_html(report_url, report_html)
    if not pdf_urls:
        raise ValueError("no PDF links found on report page")
    return pdf_urls[0]


def _build_repository(
    *,
    sqlite_path: Path,
    repository_backend: Literal["sqlite", "filesystem"],
) -> SQLiteSourceDocumentRepository | FilesystemSourceDocumentRepository:
    """Build the configured repository adapter for source versions."""
    if repository_backend == "filesystem":
        return FilesystemSourceDocumentRepository(sqlite_path)
    return SQLiteSourceDocumentRepository(sqlite_path)


def _utc_now_isoformat() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()
