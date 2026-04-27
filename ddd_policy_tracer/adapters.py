"""Persistence and sitemap adapter implementations for acquisition flows."""

from __future__ import annotations

import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from uuid import uuid4

from .domain import SourceDocumentVersion


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
                    checksum TEXT NOT NULL,
                    normalized_text TEXT NOT NULL,
                    raw_content_ref TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
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
                SELECT source_id, source_document_id, source_url, checksum,
                       normalized_text, raw_content_ref, content_type
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
                (source_id, source_document_id, source_url, checksum,
                 normalized_text, raw_content_ref, content_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version.source_id,
                    version.source_document_id,
                    version.source_url,
                    version.checksum,
                    version.normalized_text,
                    version.raw_content_ref,
                    version.content_type,
                ),
            )

    def list_versions(self, *, source_id: str) -> list[SourceDocumentVersion]:
        """List all persisted source document versions for a source."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT source_id, source_document_id, source_url, checksum,
                       normalized_text, raw_content_ref, content_type
                FROM source_document_versions
                WHERE source_id = ?
                ORDER BY id ASC
                """,
                (source_id,),
            ).fetchall()

        return [SourceDocumentVersion(*row) for row in rows]


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


def discover_urls_from_sitemap(sitemap_xml: str) -> list[str]:
    """Extract document URLs from a sitemap URL set XML payload."""
    root = ET.fromstring(sitemap_xml)
    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [
        node.text.strip()
        for node in root.findall("sm:url/sm:loc", namespace)
        if node.text
    ]
    return urls
