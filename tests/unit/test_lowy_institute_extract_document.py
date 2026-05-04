"""Unit tests for LowyInstituteSourceStrategy.extract_document.

All HTTP calls are intercepted via ``pytest-httpx``; no real network access
is performed.

The method:
1. Validates the URL is a Lowy publication detail page.
2. GETs the URL.
3. Checks HTTP 200.
4. Checks Content-Type is ``text/html``.
5. Extracts publication date (raises if absent).
6. Extracts article text and checks it is ≥ 1500 normalised characters.
7. Returns an ``ExtractedSourceDocument``.
"""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from ddd_policy_tracer.discovery.source_strategies import (
    ExtractedSourceDocument,
    LowyInstituteSourceStrategy,
    SkipSourceDocumentError,
)

# ---------------------------------------------------------------------------
# URL constants
# ---------------------------------------------------------------------------

VALID_URL = "https://www.lowyinstitute.org/publications/test-report"
NON_PUB_URL = "https://www.lowyinstitute.org/about/team"
WRONG_HOST_URL = "https://example.com/publications/test-report"

# ---------------------------------------------------------------------------
# HTML builder helpers
# ---------------------------------------------------------------------------

_LONG_TEXT = "Word " * 400  # well above the 1500-char normalised threshold


def _article_html(
    date_iso: str | None = "2024-06-01T00:00:00+00:00",
    body_text: str = _LONG_TEXT,
) -> str:
    """Build minimal Lowy publication page HTML."""
    time_tag = (
        f'<time datetime="{date_iso}">{date_iso}</time>'
        if date_iso is not None
        else ""
    )
    return (
        "<html><body>"
        "<main>"
        "<article>"
        f"{time_tag}"
        f"<p>{body_text}</p>"
        "</article>"
        "</main>"
        "</body></html>"
    )


STRATEGY = LowyInstituteSourceStrategy()


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_returns_extracted_source_document_on_success(
    httpx_mock: HTTPXMock,
) -> None:
    """Successful extraction returns an ExtractedSourceDocument."""
    html = _article_html()
    httpx_mock.add_response(
        url=VALID_URL,
        content=html.encode(),
        headers={"Content-Type": "text/html"},
    )

    result = STRATEGY.extract_document(source_url=VALID_URL)

    assert isinstance(result, ExtractedSourceDocument)
    assert result.artifact_content_type == "text/html"


def test_artifact_content_is_raw_response_bytes(httpx_mock: HTTPXMock) -> None:
    """artifact_content is the raw bytes returned by the HTTP response."""
    html = _article_html()
    encoded = html.encode()
    httpx_mock.add_response(
        url=VALID_URL,
        content=encoded,
        headers={"Content-Type": "text/html"},
    )

    result = STRATEGY.extract_document(source_url=VALID_URL)

    assert result.artifact_content == encoded


def test_published_at_is_extracted_from_datetime_attribute(
    httpx_mock: HTTPXMock,
) -> None:
    """published_at reflects the ISO date from the HTML ``datetime`` attr."""
    html = _article_html(date_iso="2024-06-01T00:00:00+00:00")
    httpx_mock.add_response(
        url=VALID_URL,
        content=html.encode(),
        headers={"Content-Type": "text/html"},
    )

    result = STRATEGY.extract_document(source_url=VALID_URL)

    assert result.published_at is not None
    assert "2024-06-01" in result.published_at


def test_extracted_text_contains_article_body(httpx_mock: HTTPXMock) -> None:
    """extracted_text includes the article paragraph content."""
    html = _article_html(body_text="Word " * 400)
    httpx_mock.add_response(
        url=VALID_URL,
        content=html.encode(),
        headers={"Content-Type": "text/html"},
    )

    result = STRATEGY.extract_document(source_url=VALID_URL)

    assert "Word" in result.extracted_text


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------


def test_raises_skip_for_non_publication_url() -> None:
    """SkipSourceDocumentError raised before any HTTP call for non-pub URLs."""
    with pytest.raises(SkipSourceDocumentError, match="not a Lowy publication"):
        STRATEGY.extract_document(source_url=NON_PUB_URL)


def test_raises_skip_for_wrong_host_url() -> None:
    """SkipSourceDocumentError raised for a URL on the wrong domain."""
    with pytest.raises(SkipSourceDocumentError, match="not a Lowy publication"):
        STRATEGY.extract_document(source_url=WRONG_HOST_URL)


def test_raises_skip_for_publications_index_url() -> None:
    """The publications listing root URL is not a valid detail page."""
    with pytest.raises(SkipSourceDocumentError, match="not a Lowy publication"):
        STRATEGY.extract_document(
            source_url="https://www.lowyinstitute.org/publications",
        )


# ---------------------------------------------------------------------------
# HTTP status errors
# ---------------------------------------------------------------------------


def test_raises_skip_when_response_is_404(httpx_mock: HTTPXMock) -> None:
    """SkipSourceDocumentError raised when the page returns HTTP 404."""
    httpx_mock.add_response(url=VALID_URL, status_code=404)

    with pytest.raises(SkipSourceDocumentError, match="404"):
        STRATEGY.extract_document(source_url=VALID_URL)


def test_raises_skip_when_response_is_500(httpx_mock: HTTPXMock) -> None:
    """SkipSourceDocumentError raised when the page returns HTTP 500."""
    httpx_mock.add_response(url=VALID_URL, status_code=500)

    with pytest.raises(SkipSourceDocumentError, match="500"):
        STRATEGY.extract_document(source_url=VALID_URL)


# ---------------------------------------------------------------------------
# Content-Type validation
# ---------------------------------------------------------------------------


def test_raises_skip_when_content_type_is_not_html(
    httpx_mock: HTTPXMock,
) -> None:
    """SkipSourceDocumentError raised when Content-Type is not text/html."""
    httpx_mock.add_response(
        url=VALID_URL,
        content=b"%PDF-1.4",
        headers={"Content-Type": "application/pdf"},
    )

    with pytest.raises(SkipSourceDocumentError, match="HTML"):
        STRATEGY.extract_document(source_url=VALID_URL)


def test_raises_skip_when_content_type_is_octet_stream(
    httpx_mock: HTTPXMock,
) -> None:
    """SkipSourceDocumentError raised for generic octet-stream content type."""
    httpx_mock.add_response(
        url=VALID_URL,
        content=b"binary",
        headers={"Content-Type": "application/octet-stream"},
    )

    with pytest.raises(SkipSourceDocumentError, match="HTML"):
        STRATEGY.extract_document(source_url=VALID_URL)


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------


def test_raises_skip_when_page_has_no_parseable_date(
    httpx_mock: HTTPXMock,
) -> None:
    """SkipSourceDocumentError raised when no date can be parsed from the page."""
    html = _article_html(date_iso=None)
    httpx_mock.add_response(
        url=VALID_URL,
        content=html.encode(),
        headers={"Content-Type": "text/html"},
    )

    with pytest.raises(SkipSourceDocumentError, match="no parseable date"):
        STRATEGY.extract_document(source_url=VALID_URL)


def test_raises_skip_when_datetime_attr_is_unparseable(
    httpx_mock: HTTPXMock,
) -> None:
    """SkipSourceDocumentError raised when datetime attr value is malformed."""
    html = (
        "<html><body><main><article>"
        '<time datetime="not-a-date">bad</time>'
        f"<p>{_LONG_TEXT}</p>"
        "</article></main></body></html>"
    )
    httpx_mock.add_response(
        url=VALID_URL,
        content=html.encode(),
        headers={"Content-Type": "text/html"},
    )

    with pytest.raises(SkipSourceDocumentError, match="no parseable date"):
        STRATEGY.extract_document(source_url=VALID_URL)


# ---------------------------------------------------------------------------
# Content length threshold
# ---------------------------------------------------------------------------


def test_raises_skip_when_extracted_text_is_below_threshold(
    httpx_mock: HTTPXMock,
) -> None:
    """SkipSourceDocumentError raised when normalised text is below 1500 chars."""
    short_text = "Short content."
    html = _article_html(body_text=short_text)
    httpx_mock.add_response(
        url=VALID_URL,
        content=html.encode(),
        headers={"Content-Type": "text/html"},
    )

    with pytest.raises(SkipSourceDocumentError, match="1500 char threshold"):
        STRATEGY.extract_document(source_url=VALID_URL)


def test_passes_when_extracted_text_is_exactly_at_threshold(
    httpx_mock: HTTPXMock,
) -> None:
    """Extraction succeeds when normalised text length equals 1500 chars exactly."""
    # Build exactly 1500 space-normalised characters: 300 × "Word" + space = 5 chars
    # "Word " * 300 → normalised length is len("Word " * 300 - trailing space) = 1499
    # Use a single word of 1500 chars to be precise.
    exact_text = "A" * 1500
    html = _article_html(body_text=exact_text)
    httpx_mock.add_response(
        url=VALID_URL,
        content=html.encode(),
        headers={"Content-Type": "text/html"},
    )

    result = STRATEGY.extract_document(source_url=VALID_URL)

    assert isinstance(result, ExtractedSourceDocument)


def test_passes_when_extracted_text_is_well_above_threshold(
    httpx_mock: HTTPXMock,
) -> None:
    """Extraction succeeds for pages with substantially more than 1500 chars."""
    html = _article_html(body_text="Word " * 1000)
    httpx_mock.add_response(
        url=VALID_URL,
        content=html.encode(),
        headers={"Content-Type": "text/html"},
    )

    result = STRATEGY.extract_document(source_url=VALID_URL)

    assert isinstance(result, ExtractedSourceDocument)
