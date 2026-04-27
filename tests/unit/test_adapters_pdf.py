"""Unit tests for report PDF link and PDF text extraction adapters."""

from io import BytesIO

from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from ddd_policy_tracer.adapters import (
    extract_pdf_urls_from_report_html,
    extract_text_from_pdf_bytes,
)


def _build_pdf_with_text(text: str) -> bytes:
    """Build a one-page PDF whose text can be extracted by pypdf."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)

    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): font_ref}
            )
        }
    )

    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)

    payload = BytesIO()
    writer.write(payload)
    return payload.getvalue()


def test_extract_pdf_urls_prefers_full_report_anchor_text() -> None:
    """Prioritize Full report links ahead of other PDF links."""
    report_url = "https://australiainstitute.org.au/report/sample/"
    html = """
    <html><body>
      <a href="/wp-content/uploads/appendix.pdf">Appendix</a>
      <a href="https://australiainstitute.org.au/wp-content/uploads/main.pdf">
        Full report
      </a>
    </body></html>
    """.strip()

    urls = extract_pdf_urls_from_report_html(report_url, html.encode("utf-8"))

    assert urls == [
        "https://australiainstitute.org.au/wp-content/uploads/main.pdf",
        "https://australiainstitute.org.au/wp-content/uploads/appendix.pdf",
    ]


def test_extract_pdf_urls_resolves_relative_urls_from_report_page() -> None:
    """Resolve relative PDF hrefs against the report page URL."""
    report_url = "https://australiainstitute.org.au/report/sample/"
    html = '<a href="../files/report.pdf">Full report</a>'

    urls = extract_pdf_urls_from_report_html(report_url, html.encode("utf-8"))

    assert urls == ["https://australiainstitute.org.au/report/files/report.pdf"]


def test_extract_text_from_pdf_bytes_returns_page_text() -> None:
    """Read plain text content from a valid one-page PDF byte payload."""
    pdf_bytes = _build_pdf_with_text("Tax reform outcomes")

    extracted = extract_text_from_pdf_bytes(pdf_bytes)

    assert "Tax reform outcomes" in extracted
