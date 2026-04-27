"""Domain primitives and normalization rules for source documents."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_QUERY_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "gclid",
    "fbclid",
}


@dataclass(frozen=True)
class SourceDocumentVersion:
    """Represent one append-only version snapshot of a source document."""

    source_id: str
    source_document_id: str
    source_url: str
    checksum: str
    normalized_text: str
    raw_content_ref: str
    content_type: str


def normalize_source_document_id(raw_url: str) -> str:
    """Normalize a source URL into a stable source-scoped document identity."""
    parts = urlsplit(raw_url.strip())
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"

    filtered_query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key in TRACKING_QUERY_KEYS:
            continue
        if value == "":
            continue
        filtered_query.append((key, value))

    query = urlencode(filtered_query, doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def compute_checksum(raw_content: bytes) -> str:
    """Compute a deterministic content hash for version change detection."""
    return sha256(raw_content).hexdigest()


def normalize_text(raw_text: str) -> str:
    """Apply minimal whitespace normalization to extracted document text."""
    return " ".join(raw_text.split())
