"""Service-layer integration tests for document extraction via ingest_source_documents.

These tests exercise the same scenarios as the unit-level extract_document
tests, but drive them through ``ingest_source_documents`` so the full
orchestration (extraction → normalisation → persistence → report) is covered.

HTTP calls are intercepted via ``pytest-httpx``; no real network access is
performed.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from pytest_httpx import HTTPXMock

from ddd_policy_tracer.discovery import ingest_source_documents
from ddd_policy_tracer.discovery.adapters import DiscoveredDocument

# ---------------------------------------------------------------------------
# Shared PDF / HTML builders
# ---------------------------------------------------------------------------


def _build_pdf(text: str = "Policy analysis content.") -> bytes:
    """Build a one-page PDF with extractable text."""
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


def _report_html(pdf_url: str, link_text: str = "Full report") -> bytes:
    """Build minimal Australia Institute report page HTML with one PDF link."""
    return (
        f'<html><body><a href="{pdf_url}">{link_text}</a></body></html>'
    ).encode()


_LONG_BODY = "Word " * 400  # well above Lowy's 1500-char threshold


def _lowy_html(
    date_iso: str | None = "2024-06-01T00:00:00+00:00",
    body_text: str = _LONG_BODY,
) -> bytes:
    """Build minimal Lowy publication page HTML."""
    time_tag = (
        f'<time datetime="{date_iso}">{date_iso}</time>'
        if date_iso is not None
        else ""
    )
    return (
        "<html><body><main><article>"
        f"{time_tag}<p>{body_text}</p>"
        "</article></main></body></html>"
    ).encode()


# ---------------------------------------------------------------------------
# Australia Institute — happy path
# ---------------------------------------------------------------------------

_AI_REPORT_URL = "https://australiainstitute.org.au/report/test/"
_AI_PDF_URL = "https://australiainstitute.org.au/wp-content/test.pdf"


def test_ai_successful_extraction_ingests_one_document(
    tmp_path: Path, httpx_mock: HTTPXMock,
) -> None:
    """A valid report page + PDF leads to one ingested document."""
    pdf_bytes = _build_pdf("Climate policy findings.")
    httpx_mock.add_response(
        url=_AI_REPORT_URL, content=_report_html(_AI_PDF_URL),
    )
    httpx_mock.add_response(
        url=_AI_PDF_URL,
        content=pdf_bytes,
        headers={"Content-Type": "application/pdf"},
    )

    report = ingest_source_documents(
        source_id="australia_institute",
        artifact_dir=tmp_path / "artifacts",
        discovered_documents=[
            DiscoveredDocument(source_url=_AI_REPORT_URL, published_at=None),
        ],
        state_path=tmp_path / "state.db",
    )

    assert report.ingested_documents == 1
    assert report.skipped_urls == 0
    assert report.failed_documents == 0
    assert report.run_status == "completed"


def test_ai_extracted_text_is_persisted(
    tmp_path: Path, httpx_mock: HTTPXMock,
) -> None:
    """Text extracted from the PDF is stored as normalized_text on the version."""
    from ddd_policy_tracer import get_source_document_versions

    pdf_bytes = _build_pdf("Energy transition report.")
    httpx_mock.add_response(
        url=_AI_REPORT_URL, content=_report_html(_AI_PDF_URL),
    )
    httpx_mock.add_response(
        url=_AI_PDF_URL,
        content=pdf_bytes,
        headers={"Content-Type": "application/pdf"},
    )
    state_path = tmp_path / "state.db"

    ingest_source_documents(
        source_id="australia_institute",
        artifact_dir=tmp_path / "artifacts",
        discovered_documents=[
            DiscoveredDocument(source_url=_AI_REPORT_URL, published_at=None),
        ],
        state_path=state_path,
    )

    versions = get_source_document_versions(
        sqlite_path=state_path, source_id="australia_institute",
    )
    assert len(versions) == 1
    assert "Energy transition report." in versions[0].normalized_text


def test_ai_published_at_falls_back_to_discovered_entry_value(
    tmp_path: Path, httpx_mock: HTTPXMock,
) -> None:
    """When strategy returns published_at=None, entry's published_at is used."""
    from ddd_policy_tracer import get_source_document_versions

    httpx_mock.add_response(
        url=_AI_REPORT_URL, content=_report_html(_AI_PDF_URL),
    )
    httpx_mock.add_response(
        url=_AI_PDF_URL,
        content=_build_pdf(),
        headers={"Content-Type": "application/pdf"},
    )
    state_path = tmp_path / "state.db"

    ingest_source_documents(
        source_id="australia_institute",
        artifact_dir=tmp_path / "artifacts",
        discovered_documents=[
            DiscoveredDocument(
                source_url=_AI_REPORT_URL,
                published_at="2025-03-15T00:00:00+00:00",
            ),
        ],
        state_path=state_path,
    )

    versions = get_source_document_versions(
        sqlite_path=state_path, source_id="australia_institute",
    )
    assert versions[0].published_at == "2025-03-15T00:00:00+00:00"


def test_ai_unchanged_checksum_is_idempotent(
    tmp_path: Path, httpx_mock: HTTPXMock,
) -> None:
    """Re-ingesting the same content does not create a duplicate version."""
    from ddd_policy_tracer import get_source_document_versions

    pdf_bytes = _build_pdf("Stable content.")
    state_path = tmp_path / "state.db"
    artifact_dir = tmp_path / "artifacts"
    docs = [DiscoveredDocument(source_url=_AI_REPORT_URL, published_at=None)]

    for _ in range(2):
        httpx_mock.add_response(
            url=_AI_REPORT_URL, content=_report_html(_AI_PDF_URL),
        )
        httpx_mock.add_response(
            url=_AI_PDF_URL,
            content=pdf_bytes,
            headers={"Content-Type": "application/pdf"},
        )

    ingest_source_documents(
        source_id="australia_institute",
        artifact_dir=artifact_dir,
        discovered_documents=docs,
        state_path=state_path,
    )
    second_report = ingest_source_documents(
        source_id="australia_institute",
        artifact_dir=artifact_dir,
        discovered_documents=docs,
        state_path=state_path,
    )

    assert second_report.ingested_documents == 0
    versions = get_source_document_versions(
        sqlite_path=state_path, source_id="australia_institute",
    )
    assert len(versions) == 1


# ---------------------------------------------------------------------------
# Australia Institute — skip / failure conditions
# ---------------------------------------------------------------------------


def test_ai_report_page_404_is_counted_as_skipped(
    tmp_path: Path, httpx_mock: HTTPXMock,
) -> None:
    """A 404 on the report page increments skipped_urls, not failed_documents."""
    httpx_mock.add_response(url=_AI_REPORT_URL, status_code=404)

    report = ingest_source_documents(
        source_id="australia_institute",
        artifact_dir=tmp_path / "artifacts",
        discovered_documents=[
            DiscoveredDocument(source_url=_AI_REPORT_URL, published_at=None),
        ],
        state_path=tmp_path / "state.db",
    )

    assert report.skipped_urls == 1
    assert report.ingested_documents == 0
    assert report.failed_documents == 0
    assert report.run_status == "completed"


def test_ai_report_page_500_is_counted_as_skipped(
    tmp_path: Path, httpx_mock: HTTPXMock,
) -> None:
    """A 500 from the report page increments skipped_urls."""
    httpx_mock.add_response(url=_AI_REPORT_URL, status_code=500)

    report = ingest_source_documents(
        source_id="australia_institute",
        artifact_dir=tmp_path / "artifacts",
        discovered_documents=[
            DiscoveredDocument(source_url=_AI_REPORT_URL, published_at=None),
        ],
        state_path=tmp_path / "state.db",
    )

    assert report.skipped_urls == 1
    assert report.run_status == "completed"


def test_ai_no_pdf_links_on_page_is_counted_as_failed(
    tmp_path: Path, httpx_mock: HTTPXMock,
) -> None:
    """No PDF links on the report page raises ValueError → failed_documents."""
    httpx_mock.add_response(
        url=_AI_REPORT_URL,
        content=b"<html><body><p>No links here.</p></body></html>",
    )

    report = ingest_source_documents(
        source_id="australia_institute",
        artifact_dir=tmp_path / "artifacts",
        discovered_documents=[
            DiscoveredDocument(source_url=_AI_REPORT_URL, published_at=None),
        ],
        state_path=tmp_path / "state.db",
    )

    assert report.failed_documents == 1
    assert report.ingested_documents == 0
    assert report.run_status == "failed"


def test_ai_pdf_wrong_content_type_is_counted_as_skipped(
    tmp_path: Path, httpx_mock: HTTPXMock,
) -> None:
    """PDF URL returning wrong Content-Type increments skipped_urls."""
    httpx_mock.add_response(
        url=_AI_REPORT_URL, content=_report_html(_AI_PDF_URL),
    )
    httpx_mock.add_response(
        url=_AI_PDF_URL,
        content=b"<html>not a pdf</html>",
        headers={"Content-Type": "text/html"},
    )

    report = ingest_source_documents(
        source_id="australia_institute",
        artifact_dir=tmp_path / "artifacts",
        discovered_documents=[
            DiscoveredDocument(source_url=_AI_REPORT_URL, published_at=None),
        ],
        state_path=tmp_path / "state.db",
    )

    assert report.skipped_urls == 1
    assert report.ingested_documents == 0


def test_ai_mixed_outcomes_reflected_in_report(
    tmp_path: Path, httpx_mock: HTTPXMock,
) -> None:
    """One success and one skip produce run_status completed."""
    ok_url = "https://australiainstitute.org.au/report/ok/"
    ok_pdf = "https://australiainstitute.org.au/wp-content/ok.pdf"
    skip_url = "https://australiainstitute.org.au/report/gone/"

    httpx_mock.add_response(url=ok_url, content=_report_html(ok_pdf))
    httpx_mock.add_response(
        url=ok_pdf,
        content=_build_pdf(),
        headers={"Content-Type": "application/pdf"},
    )
    httpx_mock.add_response(url=skip_url, status_code=410)

    report = ingest_source_documents(
        source_id="australia_institute",
        artifact_dir=tmp_path / "artifacts",
        discovered_documents=[
            DiscoveredDocument(source_url=ok_url, published_at=None),
            DiscoveredDocument(source_url=skip_url, published_at=None),
        ],
        state_path=tmp_path / "state.db",
    )

    assert report.ingested_documents == 1
    assert report.skipped_urls == 1
    assert report.run_status == "completed"


# ---------------------------------------------------------------------------
# Lowy Institute — happy path
# ---------------------------------------------------------------------------

_LOWY_URL = "https://www.lowyinstitute.org/publications/test-report"


def test_lowy_successful_extraction_ingests_one_document(
    tmp_path: Path, httpx_mock: HTTPXMock,
) -> None:
    """A valid Lowy page with date + sufficient text leads to one ingested doc."""
    httpx_mock.add_response(
        url=_LOWY_URL,
        content=_lowy_html(),
        headers={"Content-Type": "text/html"},
    )

    report = ingest_source_documents(
        source_id="lowy_institute",
        artifact_dir=tmp_path / "artifacts",
        discovered_documents=[
            DiscoveredDocument(source_url=_LOWY_URL, published_at=None),
        ],
        state_path=tmp_path / "state.db",
    )

    assert report.ingested_documents == 1
    assert report.skipped_urls == 0
    assert report.failed_documents == 0
    assert report.run_status == "completed"


def test_lowy_published_at_comes_from_html_datetime(
    tmp_path: Path, httpx_mock: HTTPXMock,
) -> None:
    """published_at stored on the version reflects the HTML datetime attribute."""
    from ddd_policy_tracer import get_source_document_versions

    httpx_mock.add_response(
        url=_LOWY_URL,
        content=_lowy_html(date_iso="2024-09-15T00:00:00+00:00"),
        headers={"Content-Type": "text/html"},
    )
    state_path = tmp_path / "state.db"

    ingest_source_documents(
        source_id="lowy_institute",
        artifact_dir=tmp_path / "artifacts",
        discovered_documents=[
            DiscoveredDocument(source_url=_LOWY_URL, published_at=None),
        ],
        state_path=state_path,
    )

    versions = get_source_document_versions(
        sqlite_path=state_path, source_id="lowy_institute",
    )
    assert "2024-09-15" in versions[0].published_at


def test_lowy_unchanged_checksum_is_idempotent(
    tmp_path: Path, httpx_mock: HTTPXMock,
) -> None:
    """Re-ingesting the same Lowy page does not create a duplicate version."""
    from ddd_policy_tracer import get_source_document_versions

    html = _lowy_html()
    state_path = tmp_path / "state.db"
    artifact_dir = tmp_path / "artifacts"
    docs = [DiscoveredDocument(source_url=_LOWY_URL, published_at=None)]

    for _ in range(2):
        httpx_mock.add_response(
            url=_LOWY_URL, content=html, headers={"Content-Type": "text/html"},
        )

    ingest_source_documents(
        source_id="lowy_institute",
        artifact_dir=artifact_dir,
        discovered_documents=docs,
        state_path=state_path,
    )
    second_report = ingest_source_documents(
        source_id="lowy_institute",
        artifact_dir=artifact_dir,
        discovered_documents=docs,
        state_path=state_path,
    )

    assert second_report.ingested_documents == 0
    versions = get_source_document_versions(
        sqlite_path=state_path, source_id="lowy_institute",
    )
    assert len(versions) == 1


# ---------------------------------------------------------------------------
# Lowy Institute — skip conditions
# ---------------------------------------------------------------------------


def test_lowy_non_publication_url_is_skipped(tmp_path: Path) -> None:
    """A non-publication Lowy URL is skipped before any HTTP call."""
    report = ingest_source_documents(
        source_id="lowy_institute",
        artifact_dir=tmp_path / "artifacts",
        discovered_documents=[
            DiscoveredDocument(
                source_url="https://www.lowyinstitute.org/about/team",
                published_at=None,
            ),
        ],
        state_path=tmp_path / "state.db",
    )

    assert report.skipped_urls == 1
    assert report.ingested_documents == 0


def test_lowy_404_is_counted_as_skipped(
    tmp_path: Path, httpx_mock: HTTPXMock,
) -> None:
    """A 404 response for a Lowy page increments skipped_urls."""
    httpx_mock.add_response(url=_LOWY_URL, status_code=404)

    report = ingest_source_documents(
        source_id="lowy_institute",
        artifact_dir=tmp_path / "artifacts",
        discovered_documents=[
            DiscoveredDocument(source_url=_LOWY_URL, published_at=None),
        ],
        state_path=tmp_path / "state.db",
    )

    assert report.skipped_urls == 1
    assert report.run_status == "completed"


def test_lowy_non_html_content_type_is_skipped(
    tmp_path: Path, httpx_mock: HTTPXMock,
) -> None:
    """A Lowy page returning non-HTML Content-Type is skipped."""
    httpx_mock.add_response(
        url=_LOWY_URL,
        content=b"%PDF",
        headers={"Content-Type": "application/pdf"},
    )

    report = ingest_source_documents(
        source_id="lowy_institute",
        artifact_dir=tmp_path / "artifacts",
        discovered_documents=[
            DiscoveredDocument(source_url=_LOWY_URL, published_at=None),
        ],
        state_path=tmp_path / "state.db",
    )

    assert report.skipped_urls == 1


def test_lowy_missing_date_on_page_is_skipped(
    tmp_path: Path, httpx_mock: HTTPXMock,
) -> None:
    """A Lowy page with no parseable date is skipped."""
    httpx_mock.add_response(
        url=_LOWY_URL,
        content=_lowy_html(date_iso=None),
        headers={"Content-Type": "text/html"},
    )

    report = ingest_source_documents(
        source_id="lowy_institute",
        artifact_dir=tmp_path / "artifacts",
        discovered_documents=[
            DiscoveredDocument(source_url=_LOWY_URL, published_at=None),
        ],
        state_path=tmp_path / "state.db",
    )

    assert report.skipped_urls == 1
    assert report.ingested_documents == 0


def test_lowy_short_content_below_threshold_is_skipped(
    tmp_path: Path, httpx_mock: HTTPXMock,
) -> None:
    """A Lowy page whose extracted text is below 1500 chars is skipped."""
    httpx_mock.add_response(
        url=_LOWY_URL,
        content=_lowy_html(body_text="Too short."),
        headers={"Content-Type": "text/html"},
    )

    report = ingest_source_documents(
        source_id="lowy_institute",
        artifact_dir=tmp_path / "artifacts",
        discovered_documents=[
            DiscoveredDocument(source_url=_LOWY_URL, published_at=None),
        ],
        state_path=tmp_path / "state.db",
    )

    assert report.skipped_urls == 1
    assert report.ingested_documents == 0
