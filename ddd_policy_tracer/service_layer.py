from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence
from uuid import uuid4

from .adapters import DiskArtifactStore, SQLiteSourceDocumentRepository, discover_urls_from_sitemap
from .domain import SourceDocumentVersion, compute_checksum, normalize_source_document_id, normalize_text


@dataclass(frozen=True)
class AcquisitionReport:
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
    run_status: Literal["completed", "completed_with_failures", "failed"] | None = None


def ingest_source_documents(
    *,
    source_id: str,
    sitemap_xml: str,
    sqlite_path: Path,
    artifact_dir: Path,
    fetch_document: Callable[..., tuple[str, bytes]],
    user_agent: str = "ddd-policy-tracer/0.1",
    is_allowed_by_robots: Callable[[str, str], bool] | None = None,
    max_retries: int = 2,
    backoff_seconds: Sequence[float] = (0.25, 0.5),
    sleep_fn: Callable[[float], None] = time.sleep,
) -> AcquisitionReport:
    repository = SQLiteSourceDocumentRepository(sqlite_path)
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

    for source_url in discover_urls_from_sitemap(sitemap_xml):
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
            extracted_text = raw_content.decode("utf-8", errors="ignore")
            normalized = normalize_text(extracted_text)
            if not normalized:
                raise ValueError("normalized_text is empty")

            checksum = compute_checksum(raw_content)
            latest_version = repository.get_latest(
                source_id=source_id,
                source_document_id=source_document_id,
            )
            if latest_version is not None and latest_version.checksum == checksum:
                continue

            raw_content_ref = artifact_store.store(
                source_document_id=source_document_id,
                content=raw_content,
            )
            version = SourceDocumentVersion(
                source_id=source_id,
                source_document_id=source_document_id,
                source_url=source_url,
                checksum=checksum,
                normalized_text=normalized,
                raw_content_ref=raw_content_ref,
                content_type=content_type,
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
        run_status: Literal["completed", "completed_with_failures", "failed"] = "completed"
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
) -> list[SourceDocumentVersion]:
    repository = SQLiteSourceDocumentRepository(sqlite_path)
    return repository.list_versions(source_id=source_id)


def _call_fetch_document(
    *,
    fetch_document: Callable[..., tuple[str, bytes]],
    source_url: str,
    user_agent: str,
) -> tuple[str, bytes]:
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
