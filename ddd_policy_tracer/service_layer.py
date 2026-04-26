from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from .adapters import DiskArtifactStore, SQLiteSourceDocumentRepository, discover_urls_from_sitemap
from .domain import SourceDocumentVersion, compute_checksum, normalize_source_document_id, normalize_text


@dataclass(frozen=True)
class AcquisitionReport:
    processed_urls: int
    ingested_documents: int
    failed_documents: int
    run_status: Literal["completed", "completed_with_failures", "failed"]


def ingest_source_documents(
    *,
    source_id: str,
    sitemap_xml: str,
    sqlite_path: Path,
    artifact_dir: Path,
    fetch_document: Callable[[str], tuple[str, bytes]],
) -> AcquisitionReport:
    repository = SQLiteSourceDocumentRepository(sqlite_path)
    artifact_store = DiskArtifactStore(artifact_dir)

    processed_urls = 0
    ingested_documents = 0
    failed_documents = 0

    for source_url in discover_urls_from_sitemap(sitemap_xml):
        processed_urls += 1
        source_document_id = normalize_source_document_id(source_url)

        try:
            content_type, raw_content = fetch_document(source_url)
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
        except Exception:
            failed_documents += 1

    if failed_documents == 0:
        run_status: Literal["completed", "completed_with_failures", "failed"] = "completed"
    elif ingested_documents == 0:
        run_status = "failed"
    else:
        run_status = "completed_with_failures"

    return AcquisitionReport(
        processed_urls=processed_urls,
        ingested_documents=ingested_documents,
        failed_documents=failed_documents,
        run_status=run_status,
    )


def get_source_document_versions(
    *,
    sqlite_path: Path,
    source_id: str,
) -> list[SourceDocumentVersion]:
    repository = SQLiteSourceDocumentRepository(sqlite_path)
    return repository.list_versions(source_id=source_id)
