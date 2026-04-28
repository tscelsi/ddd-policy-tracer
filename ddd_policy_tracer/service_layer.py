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
    DiscoveredDocument,
    DiskArtifactStore,
    FilesystemSourceDocumentRepository,
    SQLiteSourceDocumentRepository,
)
from .domain import (
    SourceDocumentVersion,
    compute_checksum,
    normalize_source_document_id,
    normalize_text,
)
from .source_strategies import SkipSourceDocumentError, get_source_strategy
from .utils.logger import get_logger

LOGGER = get_logger(__name__, ctx="service_layer")


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
    skipped_reasons: tuple[str, ...]
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
    artifact_dir: Path,
    discovered_documents: list[DiscoveredDocument],
    fetch_document: Callable[..., tuple[str, bytes]],
    state_path: Path | None = None,
    user_agent: str = "ddd-policy-tracer/0.1",
    repository_backend: Literal["sqlite", "filesystem"] = "sqlite",
    max_retries: int = 2,
    backoff_seconds: Sequence[float] = (0.25, 0.5),
    sleep_fn: Callable[[float], None] = time.sleep,
) -> AcquisitionReport:
    """Ingest discovered URLs and return aggregate acquisition outcomes."""
    repository = _build_repository(
        state_path=state_path,
        repository_backend=repository_backend,
    )
    source_strategy = get_source_strategy(source_id)
    artifact_store = DiskArtifactStore(artifact_dir)

    processed_urls = 0
    ingested_documents = 0
    failed_documents = 0
    skipped_urls = 0
    retry_attempts = 0
    document_failures: list[str] = []
    skipped_reasons: list[str] = []
    run_id = str(uuid4())
    run_logger = LOGGER.bind(run_id=run_id, source_id=source_id)
    events: list[AcquisitionEvent] = [
        AcquisitionEvent(
            event_type="AcquisitionRunStarted",
            run_id=run_id,
            source_id=source_id,
        )
    ]
    run_logger.info(
        "acquisition run started backend=%s doc_count=%s",
        repository_backend,
        len(discovered_documents),
    )

    for entry in discovered_documents:
        source_url = entry.source_url
        processed_urls += 1
        entry_logger = run_logger.bind(source_url=source_url)
        entry_logger.debug("processing discovered entry")

        source_document_id = normalize_source_document_id(source_url)
        entry_logger = entry_logger.bind(source_document_id=source_document_id)

        try:
            fetch_with_retries = _build_fetch_with_retries(
                fetch_document=fetch_document
            )
            extracted = source_strategy.extract_document(
                source_url=source_url,
                user_agent=user_agent,
                fetch_with_retries=fetch_with_retries,
                max_retries=max_retries,
                backoff_seconds=backoff_seconds,
                sleep_fn=sleep_fn,
            )
            retry_attempts += extracted.retry_attempts

            now = _utc_now_isoformat()
            normalized = normalize_text(extracted.extracted_text)
            if not normalized:
                raise ValueError("normalized_text is empty")

            checksum = compute_checksum(extracted.artifact_content)
            latest_version = repository.get_latest(
                source_id=source_id,
                source_document_id=source_document_id,
            )
            if (
                latest_version is not None
                and latest_version.checksum == checksum
            ):
                entry_logger.info("no-op: checksum unchanged")
                continue

            raw_content_ref = artifact_store.store(
                source_document_id=source_document_id,
                content=extracted.artifact_content,
            )
            version = SourceDocumentVersion(
                source_id=source_id,
                source_document_id=source_document_id,
                source_url=source_url,
                published_at=extracted.published_at or entry.published_at,
                retrieved_at=now,
                checksum=checksum,
                normalized_text=normalized,
                raw_content_ref=raw_content_ref,
                content_type=extracted.artifact_content_type,
                created_at=now,
                updated_at=now,
            )
            repository.add_version(version)
            ingested_documents += 1
            entry_logger.info(
                "ingested version content_type=%s retries=%s",
                extracted.artifact_content_type,
                extracted.retry_attempts,
            )
            events.append(
                AcquisitionEvent(
                    event_type="SourceDocumentIngested",
                    run_id=run_id,
                    source_id=source_id,
                    source_url=source_url,
                    source_document_id=source_document_id,
                )
            )
        except SkipSourceDocumentError as exc:
            skipped_urls += 1
            skipped_reasons.append(f"{source_url}: {exc}")
            entry_logger.info("skipped by source strategy: %s", exc)
            continue
        except Exception as exc:
            failed_documents += 1
            document_failures.append(f"{source_url}: {exc}")
            entry_logger.exception("document ingestion failed")
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
    run_logger.info(
        "acquisition completed processed=%s ingested=%s "
        "skipped=%s failed=%s retry_attempts=%s status=%s",
        processed_urls,
        ingested_documents,
        skipped_urls,
        failed_documents,
        retry_attempts,
        run_status,
    )

    return AcquisitionReport(
        run_id=run_id,
        processed_urls=processed_urls,
        ingested_documents=ingested_documents,
        failed_documents=failed_documents,
        skipped_urls=skipped_urls,
        retry_attempts=retry_attempts,
        document_failures=tuple(document_failures),
        skipped_reasons=tuple(skipped_reasons),
        events=tuple(events),
        run_status=run_status,
    )


def get_source_document_versions(
    *,
    state_path: Path | None = None,
    sqlite_path: Path | None = None,
    source_id: str,
    repository_backend: Literal["sqlite", "filesystem"] = "sqlite",
) -> list[SourceDocumentVersion]:
    """Load persisted source document versions for one source."""
    resolved_state_path = state_path or sqlite_path
    if resolved_state_path is None:
        raise ValueError("state_path is required")

    repository = _build_repository(
        state_path=resolved_state_path,
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


def _build_repository(
    *,
    state_path: Path,
    repository_backend: Literal["sqlite", "filesystem"],
) -> SQLiteSourceDocumentRepository | FilesystemSourceDocumentRepository:
    """Build the configured repository adapter for source versions."""
    if repository_backend == "filesystem":
        return FilesystemSourceDocumentRepository(state_path)
    return SQLiteSourceDocumentRepository(state_path)


def _utc_now_isoformat() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def _build_fetch_with_retries(
    *, fetch_document: Callable[..., tuple[str, bytes]]
) -> Callable[
    [str, str, int, Sequence[float], Callable[[float], None]],
    tuple[str, bytes, int],
]:
    """Build a source-strategy-compatible fetch wrapper with retries."""

    def fetch_with_retries(
        source_url: str,
        user_agent: str,
        max_retries: int,
        backoff_seconds: Sequence[float],
        sleep_fn: Callable[[float], None],
    ) -> tuple[str, bytes, int]:
        """Fetch one URL with retry controls provided by the caller."""
        return _fetch_with_retries(
            fetch_document=fetch_document,
            source_url=source_url,
            user_agent=user_agent,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            sleep_fn=sleep_fn,
        )

    return fetch_with_retries
