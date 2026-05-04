"""Persistence and sitemap adapter implementations for acquisition flows."""

from __future__ import annotations

import json
import sqlite3
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse
from uuid import uuid4

from bs4 import BeautifulSoup
from pypdf import PdfReader

from .domain import SourceDocumentVersion


@dataclass(frozen=True)
class DiscoveredDocument:
    """Represent one document URL entry and optional source publication time."""

    source_url: str
    published_at: str | None


LOWY_PUBLICATIONS_BASE_URL = "https://www.lowyinstitute.org/publications"


class SQLiteSourceDocumentRepository:
    """Store and retrieve source document versions from SQLite."""

    def __init__(self, sqlite_path: Path) -> None:
        """Bind the repository to a SQLite path and ensure schema readiness."""
        self._sqlite_path = sqlite_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """Open a new SQLite connection for repository operations."""
        return sqlite3.connect(self._sqlite_path)

    def _initialize(self) -> None:
        """Create repository tables when they do not already exist."""
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS source_document_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    source_document_id TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    published_at TEXT,
                    retrieved_at TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    normalized_text TEXT NOT NULL,
                    raw_content_ref TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
            )

    def get_latest(
        self,
        *,
        source_id: str,
        source_document_id: str,
    ) -> SourceDocumentVersion | None:
        """Return the latest persisted version for one source identity."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT source_id, source_document_id, source_url, published_at,
                       retrieved_at, checksum, normalized_text,
                       raw_content_ref, content_type, created_at, updated_at
                FROM source_document_versions
                WHERE source_id = ? AND source_document_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (source_id, source_document_id),
            ).fetchone()

        if row is None:
            return None

        return SourceDocumentVersion(*row)

    def add_version(self, version: SourceDocumentVersion) -> None:
        """Persist a new append-only source document version record."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO source_document_versions
                (source_id, source_document_id, source_url, published_at,
                 retrieved_at, checksum, normalized_text, raw_content_ref,
                 content_type, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version.source_id,
                    version.source_document_id,
                    version.source_url,
                    version.published_at,
                    version.retrieved_at,
                    version.checksum,
                    version.normalized_text,
                    version.raw_content_ref,
                    version.content_type,
                    version.created_at,
                    version.updated_at,
                ),
            )

    def list_versions(self, *, source_id: str) -> list[SourceDocumentVersion]:
        """List all persisted source document versions for a source."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT source_id, source_document_id, source_url, published_at,
                       retrieved_at, checksum, normalized_text,
                       raw_content_ref, content_type, created_at, updated_at
                FROM source_document_versions
                WHERE source_id = ?
                ORDER BY id ASC
                """,
                (source_id,),
            ).fetchall()

        return [SourceDocumentVersion(*row) for row in rows]


class FilesystemSourceDocumentRepository:
    """Store and retrieve source document versions from filesystem state."""

    def __init__(self, state_path: Path) -> None:
        """Bind repository state to one JSONL file path."""
        self._state_path = state_path
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

    def get_latest(
        self,
        *,
        source_id: str,
        source_document_id: str,
    ) -> SourceDocumentVersion | None:
        """Return the latest stored version for one source identity."""
        versions = self._read_all()
        matches = [
            version
            for version in versions
            if version.source_id == source_id
            and version.source_document_id == source_document_id
        ]
        if not matches:
            return None
        return matches[-1]

    def add_version(self, version: SourceDocumentVersion) -> None:
        """Append one version record to the filesystem state file."""
        record = {
            "source_id": version.source_id,
            "source_document_id": version.source_document_id,
            "source_url": version.source_url,
            "published_at": version.published_at,
            "retrieved_at": version.retrieved_at,
            "checksum": version.checksum,
            "normalized_text": version.normalized_text,
            "raw_content_ref": version.raw_content_ref,
            "content_type": version.content_type,
            "created_at": version.created_at,
            "updated_at": version.updated_at,
        }
        line = json.dumps(record, ensure_ascii=True)
        with self._state_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def list_versions(self, *, source_id: str) -> list[SourceDocumentVersion]:
        """List all stored versions for a source from filesystem state."""
        return [
            version
            for version in self._read_all()
            if version.source_id == source_id
        ]

    def _read_all(self) -> list[SourceDocumentVersion]:
        """Load all stored versions from the JSONL state file."""
        if not self._state_path.exists():
            return []

        versions: list[SourceDocumentVersion] = []
        content = self._state_path.read_text(encoding="utf-8")
        for raw_line in content.splitlines():
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            versions.append(
                SourceDocumentVersion(
                    source_id=payload["source_id"],
                    source_document_id=payload["source_document_id"],
                    source_url=payload["source_url"],
                    published_at=payload["published_at"],
                    retrieved_at=payload["retrieved_at"],
                    checksum=payload["checksum"],
                    normalized_text=payload["normalized_text"],
                    raw_content_ref=payload["raw_content_ref"],
                    content_type=payload["content_type"],
                    created_at=payload["created_at"],
                    updated_at=payload["updated_at"],
                ),
            )
        return versions


class DiskArtifactStore:
    """Persist raw fetched artifacts to local disk storage."""

    def __init__(self, artifact_dir: Path) -> None:
        """Prepare on-disk storage for raw fetched document artifacts."""
        self._artifact_dir = artifact_dir
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

    def store(self, *, source_document_id: str, content: bytes) -> str:
        """Write raw content to disk and return the artifact reference path."""
        safe_id = (
            source_document_id.replace("://", "_")
            .replace("/", "_")
            .replace("?", "_")
        )
        file_path = self._artifact_dir / f"{safe_id}_{uuid4().hex}.bin"
        file_path.write_bytes(content)
        return str(file_path)


def discover_sitemap_entries(
    sitemap_xml: str,
) -> list[DiscoveredDocument]:
    """Extract URL entries with optional publication timestamps."""
    root = ET.fromstring(sitemap_xml)
    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    entries_by_url: dict[str, DiscoveredDocument] = {}
    for node in root.findall("sm:url", namespace):
        loc_node = node.find("sm:loc", namespace)
        if loc_node is None or loc_node.text is None:
            continue

        source_url = loc_node.text.strip()
        lastmod_node = node.find("sm:lastmod", namespace)
        published_at = None
        if lastmod_node is not None and lastmod_node.text is not None:
            candidate = lastmod_node.text.strip()
            if _parse_timestamp(candidate) is not None:
                published_at = candidate

        existing = entries_by_url.get(source_url)
        if existing is None:
            entries_by_url[source_url] = DiscoveredDocument(
                source_url=source_url,
                published_at=published_at,
            )
            continue

        if _is_newer_timestamp(
            candidate_timestamp=published_at,
            current_timestamp=existing.published_at,
        ):
            entries_by_url[source_url] = DiscoveredDocument(
                source_url=source_url,
                published_at=published_at,
            )

    return list(entries_by_url.values())


def discover_urls_from_sitemap(sitemap_xml: str) -> list[str]:
    """Extract document URLs from a sitemap URL set XML payload."""
    return [entry.source_url for entry in discover_sitemap_entries(sitemap_xml)]


def discover_lowy_listing_entries(
    *,
    fetch_text_url: Callable[[str, str], str],
    user_agent: str,
    max_pages: int = 100,
    max_documents: int | None = None,
) -> tuple[list[DiscoveredDocument], int]:
    """Discover Lowy publication entries from paginated listing pages."""
    entries: list[DiscoveredDocument] = []
    pages_scanned = 0

    for page in range(max_pages):
        listing_url = f"{LOWY_PUBLICATIONS_BASE_URL}?page={page}"
        html_text = fetch_text_url(listing_url, user_agent)
        page_entries = _parse_lowy_listing_entries(html_text)
        pages_scanned += 1

        if not page_entries:
            break

        for entry in page_entries:
            if max_documents is not None and len(entries) >= max_documents:
                return entries, pages_scanned

            entries.append(entry)

    deduped = _deduplicate_entries(entries)
    return deduped, pages_scanned


def _deduplicate_entries(
    entries: list[DiscoveredDocument],
) -> list[DiscoveredDocument]:
    """Deduplicate discovered entries while retaining order."""
    deduped: list[DiscoveredDocument] = []
    seen_urls: set[str] = set()
    for entry in entries:
        if entry.source_url in seen_urls:
            continue
        seen_urls.add(entry.source_url)
        deduped.append(entry)
    return deduped


def _is_timestamp_older_than(*, timestamp: str, cutoff: datetime) -> bool:
    """Return true when one timestamp precedes the provided cutoff."""
    parsed = _parse_timestamp(timestamp)
    if parsed is None:
        return False

    if cutoff.tzinfo is None:
        normalized_cutoff = cutoff.replace(tzinfo=UTC)
    else:
        normalized_cutoff = cutoff.astimezone(UTC)
    return parsed < normalized_cutoff


def _parse_lowy_listing_entries(
    html_text: str,
) -> list[DiscoveredDocument]:
    """Parse Lowy listing HTML into publication entries."""
    soup = BeautifulSoup(html_text, "html.parser")
    entries: list[DiscoveredDocument] = []
    containers = soup.select(".card__wrapper")
    if not containers:
        containers = soup.find_all("a")

    for container in containers:
        source_url: str | None = None
        published_at: str | None = None

        preferred_anchor = container.select_one("a.card__title[href]")
        anchors = []
        if preferred_anchor is not None:
            anchors.append(preferred_anchor)
        anchors.extend(container.find_all("a", href=True))

        for anchor in anchors:
            absolute_url = urljoin(
                LOWY_PUBLICATIONS_BASE_URL,
                str(anchor.get("href")),
            )
            if _is_lowy_publication_detail_url(absolute_url):
                source_url = absolute_url
                break

        if source_url is None:
            continue

        entries.append(
            DiscoveredDocument(
                source_url=source_url,
                published_at=published_at,
            ),
        )

    return entries


def _parse_timestamp(value: str) -> datetime | None:
    """Parse one sitemap timestamp value into a comparable datetime."""
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _is_newer_timestamp(
    *, candidate_timestamp: str | None, current_timestamp: str | None,
) -> bool:
    """Return true when candidate timestamp is newer than current value."""
    candidate = (
        _parse_timestamp(candidate_timestamp)
        if candidate_timestamp is not None
        else None
    )
    current = (
        _parse_timestamp(current_timestamp)
        if current_timestamp is not None
        else None
    )

    if candidate is None:
        return False
    if current is None:
        return True
    return candidate > current


class _ReportPdfLinkParser(HTMLParser):
    """Parse report-page anchors and collect link href/text pairs."""

    def __init__(self) -> None:
        """Initialize parser state for one HTML document traversal."""
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._active_href: str | None = None
        self._active_text_parts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]],
    ) -> None:
        """Track anchor href values while entering anchor tags."""
        if tag != "a":
            return

        attr_map = dict(attrs)
        href = attr_map.get("href")
        if href is None:
            return

        self._active_href = href
        self._active_text_parts = []

    def handle_data(self, data: str) -> None:
        """Accumulate visible anchor text content for prioritization."""
        if self._active_href is not None:
            self._active_text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        """Record completed anchor link data when exiting anchor tags."""
        if tag != "a" or self._active_href is None:
            return

        text = "".join(self._active_text_parts).strip()
        self.links.append((self._active_href, text))
        self._active_href = None
        self._active_text_parts = []


def _parse_lowy_human_date(raw_text: str) -> str | None:
    """Parse Lowy human-readable date text into ISO-8601 UTC."""
    normalized = " ".join(raw_text.split())
    if not normalized:
        return None
    try:
        parsed = datetime.strptime(normalized, "%d %B %Y")
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC).isoformat()


def _is_lowy_publication_detail_url(url: str) -> bool:
    """Return true for Lowy publication detail URLs."""
    parsed = urlparse(url)
    if parsed.netloc != "www.lowyinstitute.org":
        return False
    path_parts = [part for part in parsed.path.split("/") if part]
    return len(path_parts) == 2 and path_parts[0] == "publications"


def extract_pdf_urls_from_report_html(
    report_url: str, html_bytes: bytes,
) -> list[str]:
    """Extract and prioritize absolute PDF links from a report HTML page."""
    parser = _ReportPdfLinkParser()
    parser.feed(html_bytes.decode("utf-8", errors="ignore"))

    full_report_urls: list[str] = []
    other_pdf_urls: list[str] = []

    for href, text in parser.links:
        absolute_url = urljoin(report_url, href)
        parsed = urlparse(absolute_url)
        if not parsed.path.lower().endswith(".pdf"):
            continue

        if "full report" in text.casefold():
            full_report_urls.append(absolute_url)
        else:
            other_pdf_urls.append(absolute_url)

    ordered = full_report_urls + other_pdf_urls
    return list(dict.fromkeys(ordered))


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract concatenated plain text from all pages of a PDF payload."""
    reader = PdfReader(BytesIO(pdf_bytes))
    texts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(text for text in texts if text).strip()
