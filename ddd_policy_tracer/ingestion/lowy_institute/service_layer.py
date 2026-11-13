from collections.abc import Callable
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag

from ddd_policy_tracer.ingestion.adapters import DiscoveredDocument

LOWY_BASE = "https://www.lowyinstitute.org"
LOWY_PUBLICATIONS_BASE_URL = LOWY_BASE + "/publications"
USER_AGENT: str = "ddd-policy-tracer/0.1"


def extract_document(self, *, source_url: str) -> ExtractedSourceDocument:
    """Fetch and extract Lowy publication HTML content."""
    if not _is_lowy_publication_detail_url(source_url):
        raise SkipSourceDocumentError(
            "URL is not a Lowy publication detail page",
        )
    res = httpx.get(
        source_url,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    if res.status_code != 200:
        raise SkipSourceDocumentError(
            f"Document fetch request failed: {res.status_code}",
        )
    soup = BeautifulSoup(res.text, "html.parser")
    download_tag = soup.find("a", {"class": "download-link", "href": True})
    pdf_download_link = download_tag.attrs["href"] if download_tag else None
    if pdf_download_link:
        res = httpx.get(LOWY_BASE + pdf_download_link, headers={"User-Agent": USER_AGENT}, timeout=30)
        pdf_bytes = res.content
        parser = PdfParser(min_font_size=8)

    if "text/html" not in content_type:
        raise SkipSourceDocumentError("Lowy publication page must be HTML")
    soup = BeautifulSoup(res.text, "html.parser")
    page_published_at = _extract_lowy_article_published_at(soup)
    if page_published_at is None:
        raise SkipSourceDocumentError(
            "Lowy publication page has no parseable date",
        )

    extracted_text = _extract_lowy_article_text(res.text)
    if len(_normalize_for_threshold(extracted_text)) < 1500:
        raise SkipSourceDocumentError(
            "Lowy publication page content below 1500 char threshold",
        )

    return ExtractedSourceDocument(
        file_name=source_url.split("/")[-1],
        artifact_content_type=content_type,
        artifact_content=res.content,
        extracted_text=extracted_text,
        published_at=page_published_at,
    )


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
        anchors: list[Tag] = []
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


def _is_lowy_publication_detail_url(url: str) -> bool:
    """Return true for Lowy publication detail URLs."""
    parsed = urlparse(url)
    if parsed.netloc != "www.lowyinstitute.org":
        return False
    path_parts = [part for part in parsed.path.split("/") if part]
    return len(path_parts) == 2 and path_parts[0] == "publications"


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
