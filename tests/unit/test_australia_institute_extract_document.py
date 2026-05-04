"""Unit tests for AustraliaInstituteSourceStrategy.extract_document.

All HTTP calls are intercepted via ``pytest-httpx``; no real network access
is performed.

The method:
1. GETs the report page URL.
2. Parses the HTML to find a PDF href (prioritising "Full report" anchors).
3. GETs the PDF URL.
4. Validates the Content-Type is ``application/pdf``.
5. Extracts text via pypdf and returns an ``ExtractedSourceDocument``.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from pytest_httpx import HTTPXMock

from ddd_policy_tracer.discovery.source_strategies import (
    AustraliaInstituteSourceStrategy,
    ExtractedSourceDocument,
    SkipSourceDocumentError,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

REPORT_URL = "https://australiainstitute.org.au/report/test-report/"
PDF_URL = "https://australiainstitute.org.au/wp-content/uploads/test.pdf"


def _build_pdf(text: str = "Policy content here.") -> bytes:
    """Build a minimal one-page PDF with extractable text."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        },
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})},
    )
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)

    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _report_html(pdf_href: str, link_text: str = "Full report") -> bytes:
    """Build minimal report HTML with one PDF anchor."""
    return (
        f'<html><body><a href="{pdf_href}">{link_text}</a></body></html>'
    ).encode()


STRATEGY = AustraliaInstituteSourceStrategy()


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_returns_extracted_source_document_on_success(
    httpx_mock: HTTPXMock,
) -> None:
    """Successful flow returns an ExtractedSourceDocument with PDF content."""
    pdf_bytes = _build_pdf("Test policy text.")
    httpx_mock.add_response(url=REPORT_URL, content=_report_html(PDF_URL))
    httpx_mock.add_response(
        url=PDF_URL,
        content=pdf_bytes,
        headers={"Content-Type": "application/pdf"},
    )

    result = STRATEGY.extract_document(source_url=REPORT_URL)

    assert isinstance(result, ExtractedSourceDocument)
    assert result.artifact_content_type == "application/pdf"
    assert result.artifact_content == pdf_bytes


def test_extracted_text_comes_from_pdf_content(httpx_mock: HTTPXMock) -> None:
    """extracted_text contains text extracted from PDF bytes."""
    pdf_bytes = _build_pdf("Climate policy analysis.")
    httpx_mock.add_response(url=REPORT_URL, content=_report_html(PDF_URL))
    httpx_mock.add_response(
        url=PDF_URL,
        content=pdf_bytes,
        headers={"Content-Type": "application/pdf"},
    )

    result = STRATEGY.extract_document(source_url=REPORT_URL)

    assert "Climate policy analysis." in result.extracted_text


def test_published_at_is_always_none(httpx_mock: HTTPXMock) -> None:
    """published_at is always None — the report page carries no date signal."""
    pdf_bytes = _build_pdf()
    httpx_mock.add_response(url=REPORT_URL, content=_report_html(PDF_URL))
    httpx_mock.add_response(
        url=PDF_URL,
        content=pdf_bytes,
        headers={"Content-Type": "application/pdf"},
    )

    result = STRATEGY.extract_document(source_url=REPORT_URL)

    assert result.published_at is None


def test_full_report_link_is_preferred_over_other_pdf(
    httpx_mock: HTTPXMock,
) -> None:
    """An anchor with 'Full report' text takes priority over other PDF links."""
    other_pdf = "https://australiainstitute.org.au/wp-content/appendix.pdf"
    html = (
        f'<a href="{other_pdf}">Appendix</a><a href="{PDF_URL}">Full report</a>'
    ).encode()

    pdf_bytes = _build_pdf("Full report content.")
    httpx_mock.add_response(url=REPORT_URL, content=html)
    httpx_mock.add_response(
        url=PDF_URL,
        content=pdf_bytes,
        headers={"Content-Type": "application/pdf"},
    )

    result = STRATEGY.extract_document(source_url=REPORT_URL)

    assert result.artifact_content == pdf_bytes


def test_relative_pdf_href_is_resolved_to_absolute_url(
    httpx_mock: HTTPXMock,
) -> None:
    """A relative PDF href on the report page is resolved before fetching."""
    resolved = "https://australiainstitute.org.au/wp-content/relative.pdf"
    html = b'<a href="/wp-content/relative.pdf">Full report</a>'
    pdf_bytes = _build_pdf()
    httpx_mock.add_response(url=REPORT_URL, content=html)
    httpx_mock.add_response(
        url=resolved,
        content=pdf_bytes,
        headers={"Content-Type": "application/pdf"},
    )

    result = STRATEGY.extract_document(source_url=REPORT_URL)

    assert result.artifact_content == pdf_bytes


# ---------------------------------------------------------------------------
# Error / skip conditions — report page fetch
# ---------------------------------------------------------------------------


def test_raises_skip_when_report_page_returns_non_200(
    httpx_mock: HTTPXMock,
) -> None:
    """SkipSourceDocumentError raised when report page returns HTTP 404."""
    httpx_mock.add_response(url=REPORT_URL, status_code=404)

    with pytest.raises(SkipSourceDocumentError, match="404"):
        STRATEGY.extract_document(source_url=REPORT_URL)


def test_raises_skip_when_report_page_returns_500(
    httpx_mock: HTTPXMock,
) -> None:
    """SkipSourceDocumentError raised when report page returns HTTP 500."""
    httpx_mock.add_response(url=REPORT_URL, status_code=500)

    with pytest.raises(SkipSourceDocumentError, match="500"):
        STRATEGY.extract_document(source_url=REPORT_URL)


def test_raises_value_error_when_no_pdf_links_on_page(
    httpx_mock: HTTPXMock,
) -> None:
    """ValueError raised when the report page contains no PDF anchors."""
    html = b"<html><body><p>No PDF here.</p></body></html>"
    httpx_mock.add_response(url=REPORT_URL, content=html)

    with pytest.raises(ValueError, match="no PDF links"):
        STRATEGY.extract_document(source_url=REPORT_URL)


# ---------------------------------------------------------------------------
# Error / skip conditions — PDF fetch
# ---------------------------------------------------------------------------


def test_raises_skip_when_pdf_content_type_is_not_pdf(
    httpx_mock: HTTPXMock,
) -> None:
    """SkipSourceDocumentError raised when PDF URL returns wrong content type."""
    httpx_mock.add_response(url=REPORT_URL, content=_report_html(PDF_URL))
    httpx_mock.add_response(
        url=PDF_URL,
        content=b"<html>not a pdf</html>",
        headers={"Content-Type": "text/html"},
    )

    with pytest.raises(SkipSourceDocumentError, match="text/html"):
        STRATEGY.extract_document(source_url=REPORT_URL)


def test_raises_skip_when_pdf_content_type_is_octet_stream(
    httpx_mock: HTTPXMock,
) -> None:
    """SkipSourceDocumentError raised for generic octet-stream content type."""
    httpx_mock.add_response(url=REPORT_URL, content=_report_html(PDF_URL))
    httpx_mock.add_response(
        url=PDF_URL,
        content=b"binary blob",
        headers={"Content-Type": "application/octet-stream"},
    )

    with pytest.raises(
        SkipSourceDocumentError, match="application/octet-stream",
    ):
        STRATEGY.extract_document(source_url=REPORT_URL)


def test_raises_skip_when_pdf_fetch_returns_non_200(
    httpx_mock: HTTPXMock,
) -> None:
    """SkipSourceDocumentError raised when the PDF fetch fails with a non-200 status."""
    httpx_mock.add_response(url=REPORT_URL, content=_report_html(PDF_URL))
    httpx_mock.add_response(url=PDF_URL, status_code=403)

    # The strategy does not check the PDF response status — it checks
    # Content-Type. A 403 will typically lack a PDF Content-Type and will
    # therefore raise SkipSourceDocumentError for the wrong content type.
    with pytest.raises(SkipSourceDocumentError):
        STRATEGY.extract_document(source_url=REPORT_URL)
