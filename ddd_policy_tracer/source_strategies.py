"""Source-specific discovery and extraction strategy implementations."""

from __future__ import annotations

import abc
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import urlparse

from .adapters import (
    DiscoveredDocument,
    discover_lowy_listing_entries,
    discover_sitemap_entries,
    extract_pdf_urls_from_report_html,
    extract_text_from_pdf_bytes,
)

USER_AGENT: str = "ddd-policy-tracer/0.1"


@dataclass(frozen=True)
class ExtractedSourceDocument:
    """Represent source-specific extracted content and fetch metadata."""

    artifact_content_type: str
    artifact_content: bytes
    extracted_text: str
    published_at: str | None
    retry_attempts: int


class SkipSourceDocumentError(Exception):
    """Signal that one discovered URL should be skipped, not failed."""


class AbstractSourceStrategy(abc.ABC):
    """ABC for source-specific discovery and extraction strategies.

    Exposes a common interface for the acquisition CLI to interact with
    different sources without needing to know source-specific details. Each
    strategy is responsible for implementing the discovery and extraction.
    """

    @abc.abstractmethod
    def discover_documents(
        self,
        *,
        fetch: Callable[[str, str], str],
        published_since: datetime | None,
        limit: int | None,
    ) -> tuple[list[DiscoveredDocument], int]:
        """Discover one or more document URLs from a source."""
        raise NotImplementedError

    @abc.abstractmethod
    def extract_document(
        self,
        *,
        source_url: str,
        user_agent: str,
        fetch_with_retries: Callable[
            [str, str, int, Sequence[float], Callable[[float], None]],
            tuple[str, bytes, int],
        ],
        max_retries: int,
        backoff_seconds: Sequence[float],
        sleep_fn: Callable[[float], None],
    ) -> ExtractedSourceDocument:
        """Fetch and extract one document from a source-specific URL."""
        raise NotImplementedError


@dataclass(frozen=True)
class AustraliaInstituteSourceStrategy(AbstractSourceStrategy):
    """Discover and extract Australia Institute report content from PDFs."""

    URL: str = "https://australiainstitute.org.au/sitemap_index.xml"
    CHILD_PATTERN: str = "tai_cpt_report-sitemap"

    def discover_documents(
        self,
        *,
        fetch: Callable[[str, str], str],
        published_since: datetime | None,
        limit: int | None,
    ) -> tuple[list[DiscoveredDocument], int]:
        """Discover candidate report URLs from sitemap XML."""
        selected_sitemaps = 1
        payload, selected_sitemaps = _load_sitemap_discovery_payload(
            discovery_url=self.URL,
            discovery_child_pattern=self.CHILD_PATTERN,
            user_agent=USER_AGENT,
            fetch=fetch,
        )

        entries = [
            entry
            for entry in discover_sitemap_entries(payload)
            if _is_entry_published_on_or_after(
                entry_published_at=entry.published_at,
                published_since=published_since,
            )
        ]
        if limit is not None:
            entries = entries[: max(0, limit)]
        return entries, selected_sitemaps

    def extract_document(
        self,
        *,
        source_url: str,
        user_agent: str,
        fetch_with_retries: Callable[
            [str, str, int, Sequence[float], Callable[[float], None]],
            tuple[str, bytes, int],
        ],
        max_retries: int,
        backoff_seconds: Sequence[float],
        sleep_fn: Callable[[float], None],
    ) -> ExtractedSourceDocument:
        """Fetch report page and selected PDF, then extract PDF text."""
        content_type, raw_content, used_retries = fetch_with_retries(
            source_url,
            user_agent,
            max_retries,
            backoff_seconds,
            sleep_fn,
        )
        if content_type != "text/html":
            raise ValueError("report page must be HTML")

        pdf_url = _select_pdf_url_from_report_html(
            report_url=source_url,
            report_html=raw_content,
        )
        pdf_content_type, pdf_content, pdf_retries = fetch_with_retries(
            pdf_url,
            user_agent,
            max_retries,
            backoff_seconds,
            sleep_fn,
        )
        if pdf_content_type != "application/pdf":
            raise ValueError("selected report file is not a PDF")

        return ExtractedSourceDocument(
            artifact_content_type=pdf_content_type,
            artifact_content=pdf_content,
            extracted_text=extract_text_from_pdf_bytes(pdf_content),
            published_at=None,
            retry_attempts=used_retries + pdf_retries,
        )


def get_source_strategy(
    source_id: str,
) -> AbstractSourceStrategy:
    """Resolve the configured source strategy or raise for unsupported IDs."""
    if source_id == "australia_institute":
        return AustraliaInstituteSourceStrategy()
    if source_id == "lowy_institute":
        return LowyInstituteSourceStrategy()
    raise ValueError(f"Unsupported source_id: {source_id}")


@dataclass(frozen=True)
class LowyInstituteSourceStrategy(AbstractSourceStrategy):
    """Discover Lowy listing entries while extraction is not yet implemented."""

    def discover_documents(
        self,
        *,
        fetch: Callable[[str, str], str],
        published_since: datetime | None,
        limit: int | None,
    ) -> tuple[list[DiscoveredDocument], int]:
        """Discover Lowy publication entries from listing pages."""
        return discover_lowy_listing_entries(
            fetch=fetch,
            user_agent=USER_AGENT,
            published_since=published_since,
            max_documents=limit,
        )

    def extract_document(
        self,
        *,
        source_url: str,
        user_agent: str,
        fetch_with_retries: Callable[
            [str, str, int, Sequence[float], Callable[[float], None]],
            tuple[str, bytes, int],
        ],
        max_retries: int,
        backoff_seconds: Sequence[float],
        sleep_fn: Callable[[float], None],
    ) -> ExtractedSourceDocument:
        """Fetch and extract Lowy publication HTML content."""
        if not _is_lowy_publication_detail_url(source_url):
            raise SkipSourceDocumentError(
                "URL is not a Lowy publication detail page"
            )

        content_type, raw_content, retry_attempts = fetch_with_retries(
            source_url,
            user_agent,
            max_retries,
            backoff_seconds,
            sleep_fn,
        )
        if content_type != "text/html":
            raise SkipSourceDocumentError("Lowy publication page must be HTML")

        html_text = raw_content.decode("utf-8", errors="ignore")
        page_published_at = _extract_lowy_article_published_at(html_text)
        if page_published_at is None:
            raise SkipSourceDocumentError(
                "Lowy publication page has no parseable date"
            )

        extracted_text = _extract_lowy_article_text(html_text)
        if len(_normalize_for_threshold(extracted_text)) < 1500:
            raise SkipSourceDocumentError(
                "Lowy publication page content below 1500 char threshold"
            )

        return ExtractedSourceDocument(
            artifact_content_type="text/html",
            artifact_content=raw_content,
            extracted_text=extracted_text,
            published_at=page_published_at,
            retry_attempts=retry_attempts,
        )


def _select_pdf_url_from_report_html(
    *, report_url: str, report_html: bytes
) -> str:
    """Choose one PDF URL from a report page or raise when none exist."""
    pdf_urls = extract_pdf_urls_from_report_html(report_url, report_html)
    if not pdf_urls:
        raise ValueError("no PDF links found on report page")
    return pdf_urls[0]


def _extract_lowy_article_published_at(html_text: str) -> str | None:
    """Extract one parseable publication timestamp from a Lowy page."""
    time_match = re.search(
        r"<time[^>]*datetime=[\"']([^\"']+)[\"']",
        html_text,
        flags=re.IGNORECASE,
    )
    if time_match is not None:
        parsed = _parse_iso_timestamp(time_match.group(1).strip())
        if parsed is not None:
            return parsed.isoformat()

    date_match = re.search(
        r"\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b",
        html_text,
    )
    if date_match is None:
        return None

    return _parse_human_date(date_match.group(1))


def _parse_iso_timestamp(value: str) -> datetime | None:
    """Parse one ISO-8601 timestamp value."""
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _parse_human_date(value: str) -> str | None:
    """Parse one human-readable date string into ISO-8601."""
    normalized = " ".join(value.split())
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(normalized, fmt).isoformat()
        except ValueError:
            continue
    return None


def _extract_lowy_article_text(html_text: str) -> str:
    """Extract core Lowy publication text and drop acknowledgements."""
    parser = _LowyArticleTextParser()
    parser.feed(html_text)
    combined = "\n".join(segment for segment in parser.segments if segment)
    return _drop_acknowledgements_section(combined)


def _normalize_for_threshold(value: str) -> str:
    """Normalize text for deterministic qualification length checks."""
    return " ".join(value.split())


def _drop_acknowledgements_section(value: str) -> str:
    """Remove acknowledgements heading and trailing section content."""
    lines = [line.strip() for line in value.splitlines()]
    output_lines: list[str] = []
    dropping = False
    for line in lines:
        if not dropping and line.casefold() == "acknowledgements":
            dropping = True
            continue
        if not dropping:
            output_lines.append(line)
    return "\n".join(line for line in output_lines if line)


class _LowyArticleTextParser(HTMLParser):
    """Extract human-readable text from Lowy publication article markup."""

    _BLOCK_TAGS = {
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "li",
        "section",
        "div",
        "br",
    }
    _IGNORE_TAGS = {"script", "style", "nav", "header", "footer", "aside"}

    def __init__(self) -> None:
        """Initialize parser state for one article document."""
        super().__init__()
        self.segments: list[str] = []
        self._main_depth = 0
        self._article_depth = 0
        self._ignore_depth = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        """Track content container depth and capture block boundaries."""
        if tag == "main":
            self._main_depth += 1
        if tag == "article":
            self._article_depth += 1
        if tag in self._IGNORE_TAGS:
            self._ignore_depth += 1
            return

        if not self._is_inside_content_root() or self._ignore_depth > 0:
            return
        if tag in self._BLOCK_TAGS:
            self.segments.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Update parser depth markers while leaving tags."""
        if tag in self._IGNORE_TAGS and self._ignore_depth > 0:
            self._ignore_depth -= 1
            return
        if tag == "main" and self._main_depth > 0:
            self._main_depth -= 1
        if tag == "article" and self._article_depth > 0:
            self._article_depth -= 1

    def handle_data(self, data: str) -> None:
        """Capture textual content from the main article container."""
        if not self._is_inside_content_root() or self._ignore_depth > 0:
            return

        normalized = " ".join(data.split())
        if normalized:
            self.segments.append(normalized)

    def _is_inside_content_root(self) -> bool:
        """Return true when parser is within main/article content."""
        return self._main_depth > 0 or self._article_depth > 0


def _is_lowy_publication_detail_url(url: str) -> bool:
    """Return true when one URL is a Lowy publication detail page."""
    parsed = urlparse(url)
    if parsed.netloc not in {"www.lowyinstitute.org", "lowyinstitute.org"}:
        return False
    path = parsed.path.strip("/")
    if not path.startswith("publications/"):
        return False
    return len(path.split("/")) == 2


def _is_entry_published_on_or_after(
    *, entry_published_at: str | None, published_since: datetime | None
) -> bool:
    """Return true when an entry passes the optional publish-time filter."""
    if published_since is None:
        return True
    if entry_published_at is None:
        return False

    entry_timestamp = _parse_iso_timestamp(entry_published_at)
    if entry_timestamp is None:
        return False
    return entry_timestamp >= published_since


def _load_sitemap_discovery_payload(
    *,
    discovery_url: str,
    discovery_child_pattern: str | None,
    user_agent: str,
    fetch: Callable[[str, str], str],
) -> tuple[str, int]:
    """Load discovery payload from local file or sitemap URL sources."""
    root_xml = fetch(discovery_url, user_agent)
    if _is_sitemap_index(root_xml):
        child_sitemaps = _discover_child_sitemaps(root_xml)
        if discovery_child_pattern is not None:
            child_sitemaps = [
                url for url in child_sitemaps if discovery_child_pattern in url
            ]
        child_urlsets = [fetch(url, user_agent) for url in child_sitemaps]
        return _merge_urlsets(child_urlsets), len(child_sitemaps)

    return root_xml, 1


def _is_sitemap_index(xml_text: str) -> bool:
    """Return true when one XML payload is a sitemap index."""
    root = ET.fromstring(xml_text)
    return root.tag.endswith("sitemapindex")


def _discover_child_sitemaps(sitemap_index_xml: str) -> list[str]:
    """Extract child sitemap URLs from one sitemap index payload."""
    root = ET.fromstring(sitemap_index_xml)
    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [
        node.text.strip()
        for node in root.findall("sm:sitemap/sm:loc", namespace)
        if node.text
    ]


def _merge_urlsets(urlset_xml_documents: list[str]) -> str:
    """Merge URL set XML documents into one deduplicated payload."""
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
    lines.extend(
        [
            "</urlset>",
        ]
    )
    return "\n".join(lines)
