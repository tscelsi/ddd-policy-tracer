"""Integration tests for core acquisition service behavior."""

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from ddd_policy_tracer import (
    get_source_document_versions,
    ingest_source_documents,
)
from ddd_policy_tracer.adapters import DiscoveredSitemapEntry


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


def _report_html_with_pdf_link(pdf_url: str) -> bytes:
    """Build report HTML content with one Full report PDF anchor."""
    html = (
        '<a href="https://australiainstitute.org.au/about">About</a>'
        f'<a href="{pdf_url}">Full report</a>'
    )
    return html.encode("utf-8")


def test_ingest_from_sitemap_persists_first_source_document_version(
    tmp_path: Path,
) -> None:
    """Verify one sitemap URL is ingested and persisted with artifact data."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url>
        <loc>https://australiainstitute.org.au/report-1?utm_source=newsletter</loc>
        <lastmod>2026-04-24T10:12:30+00:00</lastmod>
      </url>
    </urlset>
    """.strip()

    report_url = (
        "https://australiainstitute.org.au/report-1?utm_source=newsletter"
    )
    pdf_url = "https://australiainstitute.org.au/wp-content/report-1.pdf"

    def fetch_document(url: str) -> tuple[str, bytes]:
        if url == report_url:
            return "text/html", _report_html_with_pdf_link(pdf_url)
        if url == pdf_url:
            return (
                "application/pdf",
                _build_pdf_with_text("This is a report about policy reform."),
            )
        raise AssertionError(f"unexpected url fetched: {url}")

    report = ingest_source_documents(
        source_id="australia_institute",
        sitemap_xml=sitemap_xml,
        sqlite_path=sqlite_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
    )

    assert report.processed_urls == 1
    assert report.ingested_documents == 1
    assert report.failed_documents == 0
    assert report.run_status == "completed"

    versions = get_source_document_versions(
        sqlite_path=sqlite_path,
        source_id="australia_institute",
    )

    assert len(versions) == 1
    version = versions[0]
    assert (
        version.source_document_id
        == "https://australiainstitute.org.au/report-1"
    )
    assert version.published_at == "2026-04-24T10:12:30+00:00"
    assert version.retrieved_at
    assert version.normalized_text == "This is a report about policy reform."
    assert version.raw_content_ref.startswith(str(artifact_dir))
    assert Path(version.raw_content_ref).exists()
    assert version.created_at
    assert version.updated_at


def test_ingest_rejects_unknown_source_id(tmp_path: Path) -> None:
    """Raise a clear error for unsupported source identifiers."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://example.org/report</loc></url>
    </urlset>
    """.strip()

    def fetch_document(_: str) -> tuple[str, bytes]:
        return "text/plain", b"unused"

    try:
        ingest_source_documents(
            source_id="unknown_source",
            sitemap_xml=sitemap_xml,
            sqlite_path=sqlite_path,
            artifact_dir=artifact_dir,
            fetch_document=fetch_document,
        )
    except ValueError as exc:
        assert "Unsupported source_id" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported source")


def test_reprocessing_same_unchanged_url_is_idempotent(tmp_path: Path) -> None:
    """Ensure unchanged content does not create duplicate persisted versions."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://australiainstitute.org.au/report-1?utm_source=newsletter</loc></url>
    </urlset>
    """.strip()

    report_url = (
        "https://australiainstitute.org.au/report-1?utm_source=newsletter"
    )
    pdf_url = "https://australiainstitute.org.au/wp-content/report-1.pdf"

    def fetch_document(url: str) -> tuple[str, bytes]:
        if url == report_url:
            return "text/html", _report_html_with_pdf_link(pdf_url)
        if url == pdf_url:
            return (
                "application/pdf",
                _build_pdf_with_text("Unchanged report body."),
            )
        raise AssertionError(f"unexpected url fetched: {url}")

    first_report = ingest_source_documents(
        source_id="australia_institute",
        sitemap_xml=sitemap_xml,
        sqlite_path=sqlite_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
    )
    second_report = ingest_source_documents(
        source_id="australia_institute",
        sitemap_xml=sitemap_xml,
        sqlite_path=sqlite_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
    )

    assert first_report.ingested_documents == 1
    assert second_report.ingested_documents == 0

    versions = get_source_document_versions(
        sqlite_path=sqlite_path, source_id="australia_institute"
    )
    assert len(versions) == 1


def test_published_since_filters_out_old_sitemap_entries(
    tmp_path: Path,
) -> None:
    """Skip entries whose sitemap lastmod predates published_since."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url>
        <loc>https://australiainstitute.org.au/report/new</loc>
        <lastmod>2026-02-10T00:00:00+00:00</lastmod>
      </url>
      <url>
        <loc>https://australiainstitute.org.au/report/old</loc>
        <lastmod>2020-01-01T00:00:00+00:00</lastmod>
      </url>
    </urlset>
    """.strip()

    fetch_calls: list[str] = []

    def fetch_document(url: str) -> tuple[str, bytes]:
        fetch_calls.append(url)
        if url.endswith("/report/new"):
            return (
                "text/html",
                _report_html_with_pdf_link(
                    "https://australiainstitute.org.au/wp-content/new.pdf"
                ),
            )
        if url.endswith("/new.pdf"):
            return "application/pdf", _build_pdf_with_text("Fresh content")
        raise AssertionError(f"unexpected url fetched: {url}")

    report = ingest_source_documents(
        source_id="australia_institute",
        sitemap_xml=sitemap_xml,
        sqlite_path=sqlite_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
        published_since=datetime(2025, 1, 1, tzinfo=UTC),
    )

    assert report.processed_urls == 2
    assert report.ingested_documents == 1
    assert report.skipped_urls == 1
    assert fetch_calls == [
        "https://australiainstitute.org.au/report/new",
        "https://australiainstitute.org.au/wp-content/new.pdf",
    ]


def test_checksum_change_appends_new_version(tmp_path: Path) -> None:
    """Ensure checksum changes append a new version for the same identity."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://australiainstitute.org.au/report-1</loc></url>
    </urlset>
    """.strip()

    payloads = [
        _build_pdf_with_text("Version 1 content."),
        _build_pdf_with_text("Version 2 content with updates."),
    ]
    report_url = "https://australiainstitute.org.au/report-1"
    pdf_url = "https://australiainstitute.org.au/wp-content/report-1.pdf"

    def fetch_document(url: str) -> tuple[str, bytes]:
        if url == report_url:
            return "text/html", _report_html_with_pdf_link(pdf_url)
        if url == pdf_url:
            return "application/pdf", payloads.pop(0)
        raise AssertionError(f"unexpected url fetched: {url}")

    ingest_source_documents(
        source_id="australia_institute",
        sitemap_xml=sitemap_xml,
        sqlite_path=sqlite_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
    )
    second_report = ingest_source_documents(
        source_id="australia_institute",
        sitemap_xml=sitemap_xml,
        sqlite_path=sqlite_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
    )

    assert second_report.ingested_documents == 1

    versions = get_source_document_versions(
        sqlite_path=sqlite_path, source_id="australia_institute"
    )
    assert len(versions) == 2
    assert versions[0].checksum != versions[1].checksum
    assert versions[0].published_at is None


def test_run_status_is_completed_with_failures_for_mixed_outcomes(
    tmp_path: Path,
) -> None:
    """Report mixed run outcomes as completed_with_failures."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://australiainstitute.org.au/report-ok</loc></url>
      <url><loc>https://australiainstitute.org.au/report-fail</loc></url>
    </urlset>
    """.strip()

    def fetch_document(url: str) -> tuple[str, bytes]:
        if url.endswith("/report-fail"):
            raise RuntimeError("temporary fetch failure")
        if url.endswith("/report-ok"):
            return (
                "text/html",
                _report_html_with_pdf_link(
                    "https://australiainstitute.org.au/wp-content/report-ok.pdf"
                ),
            )
        if url.endswith("/report-ok.pdf"):
            return (
                "application/pdf",
                _build_pdf_with_text("Healthy document content."),
            )
        raise AssertionError(f"unexpected url fetched: {url}")

    report = ingest_source_documents(
        source_id="australia_institute",
        sitemap_xml=sitemap_xml,
        sqlite_path=sqlite_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
    )

    assert report.processed_urls == 2
    assert report.ingested_documents == 1
    assert report.failed_documents == 1
    assert report.run_status == "completed_with_failures"


def test_run_status_is_failed_when_all_urls_fail(tmp_path: Path) -> None:
    """Report run status failed when every processed URL fails."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://australiainstitute.org.au/report-a</loc></url>
      <url><loc>https://australiainstitute.org.au/report-b</loc></url>
    </urlset>
    """.strip()

    def fetch_document(_: str) -> tuple[str, bytes]:
        raise RuntimeError("network unavailable")

    report = ingest_source_documents(
        source_id="australia_institute",
        sitemap_xml=sitemap_xml,
        sqlite_path=sqlite_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
    )

    assert report.processed_urls == 2
    assert report.ingested_documents == 0
    assert report.failed_documents == 2
    assert report.run_status == "failed"


def test_compliance_blocks_disallowed_url_and_uses_configured_user_agent(
    tmp_path: Path,
) -> None:
    """Skip disallowed URLs and pass user-agent into compliant fetch calls."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://australiainstitute.org.au/allowed-report</loc></url>
      <url><loc>https://australiainstitute.org.au/blocked-report</loc></url>
    </urlset>
    """.strip()

    fetched_urls: list[str] = []
    seen_user_agents: list[str] = []

    def fetch_document(url: str, user_agent: str) -> tuple[str, bytes]:
        fetched_urls.append(url)
        seen_user_agents.append(user_agent)
        if url.endswith("/allowed-report"):
            return (
                "text/html",
                _report_html_with_pdf_link(
                    "https://australiainstitute.org.au/wp-content/allowed.pdf"
                ),
            )
        if url.endswith("/allowed.pdf"):
            return "application/pdf", _build_pdf_with_text("Compliant content.")
        raise AssertionError(f"unexpected url fetched: {url}")

    def is_allowed_by_robots(url: str, _: str) -> bool:
        return "blocked" not in url

    report = ingest_source_documents(
        source_id="australia_institute",
        sitemap_xml=sitemap_xml,
        sqlite_path=sqlite_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
        user_agent="policy-tracer/0.1",
        is_allowed_by_robots=is_allowed_by_robots,
    )

    assert fetched_urls == [
        "https://australiainstitute.org.au/allowed-report",
        "https://australiainstitute.org.au/wp-content/allowed.pdf",
    ]
    assert seen_user_agents == ["policy-tracer/0.1", "policy-tracer/0.1"]
    assert report.processed_urls == 2
    assert report.ingested_documents == 1
    assert report.failed_documents == 0
    assert report.skipped_urls == 1


def test_transient_failure_is_retried_and_observable(tmp_path: Path) -> None:
    """Retry transient fetch failures and expose retry attempts in report."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://australiainstitute.org.au/retry-report</loc></url>
    </urlset>
    """.strip()

    attempts = 0

    def fetch_document(url: str) -> tuple[str, bytes]:
        nonlocal attempts
        if url.endswith("/retry-report"):
            attempts += 1
            if attempts < 3:
                raise TimeoutError("temporary timeout")
            return (
                "text/html",
                _report_html_with_pdf_link(
                    "https://australiainstitute.org.au/wp-content/retry.pdf"
                ),
            )
        if url.endswith("/retry.pdf"):
            return (
                "application/pdf",
                _build_pdf_with_text("Recovered after retries."),
            )
        raise AssertionError(f"unexpected url fetched: {url}")

    sleep_calls: list[float] = []

    report = ingest_source_documents(
        source_id="australia_institute",
        sitemap_xml=sitemap_xml,
        sqlite_path=sqlite_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
        max_retries=2,
        backoff_seconds=(0.1, 0.2),
        sleep_fn=sleep_calls.append,
    )

    assert attempts == 3
    assert sleep_calls == [0.1, 0.2]
    assert report.ingested_documents == 1
    assert report.failed_documents == 0
    assert report.retry_attempts == 2


def test_terminal_failure_records_actionable_reason_after_retries_exhausted(
    tmp_path: Path,
) -> None:
    """Record actionable failure details after retries are exhausted."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://australiainstitute.org.au/fails-hard</loc></url>
    </urlset>
    """.strip()

    attempts = 0

    def fetch_document(_: str) -> tuple[str, bytes]:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("request timeout")

    report = ingest_source_documents(
        source_id="australia_institute",
        sitemap_xml=sitemap_xml,
        sqlite_path=sqlite_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
        max_retries=1,
        backoff_seconds=(0.01,),
        sleep_fn=lambda _seconds: None,
    )

    assert attempts == 2
    assert report.ingested_documents == 0
    assert report.failed_documents == 1
    assert report.run_status == "failed"
    assert report.retry_attempts == 0
    assert len(report.document_failures) == 1
    assert "fails-hard" in report.document_failures[0]
    assert "request timeout" in report.document_failures[0]


def test_domain_events_are_emitted_in_expected_lifecycle_sequence(
    tmp_path: Path,
) -> None:
    """Emit ordered run/document lifecycle events with run context."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://australiainstitute.org.au/event-ok</loc></url>
      <url><loc>https://australiainstitute.org.au/event-fail</loc></url>
    </urlset>
    """.strip()

    def fetch_document(url: str) -> tuple[str, bytes]:
        if url.endswith("/event-fail"):
            raise RuntimeError("boom")
        if url.endswith("/event-ok"):
            return (
                "text/html",
                _report_html_with_pdf_link(
                    "https://australiainstitute.org.au/wp-content/event-ok.pdf"
                ),
            )
        if url.endswith("/event-ok.pdf"):
            return "application/pdf", _build_pdf_with_text("Eventful content")
        raise AssertionError(f"unexpected url fetched: {url}")

    report = ingest_source_documents(
        source_id="australia_institute",
        sitemap_xml=sitemap_xml,
        sqlite_path=sqlite_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
    )

    assert [event.event_type for event in report.events] == [
        "AcquisitionRunStarted",
        "SourceDocumentIngested",
        "SourceDocumentIngestionFailed",
        "AcquisitionRunCompleted",
    ]
    assert all(event.run_id == report.run_id for event in report.events)
    assert all(
        event.source_id == "australia_institute" for event in report.events
    )
    assert (
        report.events[1].source_url
        == "https://australiainstitute.org.au/event-ok"
    )
    assert (
        report.events[2].source_url
        == "https://australiainstitute.org.au/event-fail"
    )
    assert report.events[3].run_status == "completed_with_failures"


def test_ingest_report_page_downloads_pdf_and_persists_pdf_text(
    tmp_path: Path,
) -> None:
    """Ingest only extracted PDF text for report-page sitemap URLs."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://australiainstitute.org.au/report/example/</loc></url>
    </urlset>
    """.strip()

    pdf_url = "https://australiainstitute.org.au/wp-content/report.pdf"
    pdf_bytes = _build_pdf_with_text("Budget policy report")

    def fetch_document(url: str) -> tuple[str, bytes]:
        if url.endswith("/report/example/"):
            return "text/html", _report_html_with_pdf_link(pdf_url)
        if url == pdf_url:
            return "application/pdf", pdf_bytes
        raise AssertionError(f"unexpected url fetched: {url}")

    report = ingest_source_documents(
        source_id="australia_institute",
        sitemap_xml=sitemap_xml,
        sqlite_path=sqlite_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
    )

    assert report.ingested_documents == 1

    versions = get_source_document_versions(
        sqlite_path=sqlite_path,
        source_id="australia_institute",
    )
    assert len(versions) == 1
    assert "Budget policy report" in versions[0].normalized_text


def test_ingest_fails_when_report_page_has_no_pdf_link(tmp_path: Path) -> None:
    """Fail report ingestion when no PDF link exists on the report page."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://australiainstitute.org.au/report/no-pdf/</loc></url>
    </urlset>
    """.strip()

    def fetch_document(url: str) -> tuple[str, bytes]:
        assert url.endswith("/report/no-pdf/")
        return "text/html", b"<html><body>No downloadable file</body></html>"

    report = ingest_source_documents(
        source_id="australia_institute",
        sitemap_xml=sitemap_xml,
        sqlite_path=sqlite_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
    )

    assert report.processed_urls == 1
    assert report.ingested_documents == 0
    assert report.failed_documents == 1
    assert "no PDF links found on report page" in report.document_failures[0]


def test_filesystem_repository_persists_versions_without_sqlite(
    tmp_path: Path,
) -> None:
    """Persist versions in filesystem backend when sqlite is not used."""
    state_path = tmp_path / "acquisition.jsonl"
    artifact_dir = tmp_path / "artifacts"
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://australiainstitute.org.au/report/fs/</loc></url>
    </urlset>
    """.strip()

    pdf_url = "https://australiainstitute.org.au/wp-content/fs-report.pdf"

    def fetch_document(url: str) -> tuple[str, bytes]:
        if url.endswith("/report/fs/"):
            return "text/html", _report_html_with_pdf_link(pdf_url)
        if url == pdf_url:
            return "application/pdf", _build_pdf_with_text("Filesystem content")
        raise AssertionError(f"unexpected url fetched: {url}")

    report = ingest_source_documents(
        source_id="australia_institute",
        sitemap_xml=sitemap_xml,
        sqlite_path=state_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
        repository_backend="filesystem",
    )

    assert report.ingested_documents == 1
    assert state_path.exists()

    versions = get_source_document_versions(
        sqlite_path=state_path,
        source_id="australia_institute",
        repository_backend="filesystem",
    )
    assert len(versions) == 1
    assert "Filesystem content" in versions[0].normalized_text


def test_lowy_ingests_html_publication_content_and_stores_html_artifact(
    tmp_path: Path,
) -> None:
    """Persist Lowy publication HTML content when quality gates pass."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    discovered = [
        DiscoveredSitemapEntry(
            source_url="https://www.lowyinstitute.org/publications/sea-security",
            published_at="2025-08-19T00:00:00+00:00",
        )
    ]
    publication_html = """
    <html>
      <main>
        <h1>Southeast Asia's evolving defence partnerships</h1>
        <time datetime="2025-08-19T00:00:00+00:00">19 August 2025</time>
        <p>Key Findings</p>
        <p>{long_text}</p>
        <h2>References</h2>
        <p>[1] Evidence and context.</p>
      </main>
    </html>
    """.strip().format(long_text=("Important sentence. " * 120))

    def fetch_document(url: str) -> tuple[str, bytes]:
        assert url.endswith("/publications/sea-security")
        return "text/html", publication_html.encode("utf-8")

    report = ingest_source_documents(
        source_id="lowy_institute",
        sitemap_xml="",
        sqlite_path=sqlite_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
        discovered_entries=discovered,
    )

    assert report.processed_urls == 1
    assert report.ingested_documents == 1
    assert report.failed_documents == 0
    assert report.skipped_urls == 0

    versions = get_source_document_versions(
        sqlite_path=sqlite_path,
        source_id="lowy_institute",
    )
    assert len(versions) == 1
    version = versions[0]
    assert version.source_document_id.endswith("/publications/sea-security")
    assert version.content_type == "text/html"
    assert "Key Findings" in version.normalized_text
    assert "References" in version.normalized_text
    assert Path(version.raw_content_ref).read_bytes().startswith(b"<html>")


def test_lowy_skips_page_with_short_content_or_missing_date(
    tmp_path: Path,
) -> None:
    """Skip Lowy pages that fail article-date/length quality gates."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    shim_entries = [
        DiscoveredSitemapEntry(
            source_url="https://www.lowyinstitute.org/publications/short",
            published_at="2025-08-19T00:00:00+00:00",
        ),
        DiscoveredSitemapEntry(
            source_url="https://www.lowyinstitute.org/publications/no-date",
            published_at="2025-08-20T00:00:00+00:00",
        ),
    ]

    def fetch_document(url: str) -> tuple[str, bytes]:
        if url.endswith("/short"):
            return (
                "text/html",
                (
                    b"<html><main><time datetime='2025-08-19T00:00:00+00:00'>"
                    b"</time><p>short</p></main></html>"
                ),
            )
        if url.endswith("/no-date"):
            return (
                "text/html",
                b"<html><main><p>" + (b"text " * 700) + b"</p></main></html>",
            )
        raise AssertionError(f"unexpected url fetched: {url}")

    report = ingest_source_documents(
        source_id="lowy_institute",
        sitemap_xml="",
        sqlite_path=sqlite_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
        discovered_entries=shim_entries,
    )

    assert report.processed_urls == 2
    assert report.ingested_documents == 0
    assert report.failed_documents == 0
    assert report.skipped_urls == 2


def test_lowy_drops_acknowledgements_from_normalized_text(
    tmp_path: Path,
) -> None:
    """Exclude acknowledgements section while keeping core report content."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    entries = [
        DiscoveredSitemapEntry(
            source_url="https://www.lowyinstitute.org/publications/with-acks",
            published_at="2025-08-19T00:00:00+00:00",
        )
    ]

    html_bytes = (
        b"<html><main><time datetime='2025-08-19T00:00:00+00:00'></time>"
        b"<h1>Title</h1><p>"
        + (b"Core analysis text. " * 120)
        + b"</p><h2>Acknowledgements</h2><p>Team thanks everyone.</p>"
        b"</main></html>"
    )

    def fetch_document(url: str) -> tuple[str, bytes]:
        assert url.endswith("/with-acks")
        return "text/html", html_bytes

    report = ingest_source_documents(
        source_id="lowy_institute",
        sitemap_xml="",
        sqlite_path=sqlite_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
        discovered_entries=entries,
    )

    assert report.ingested_documents == 1

    versions = get_source_document_versions(
        sqlite_path=sqlite_path,
        source_id="lowy_institute",
    )
    assert len(versions) == 1
    normalized_text = versions[0].normalized_text
    assert "Core analysis text." in normalized_text
    assert "Team thanks everyone." not in normalized_text


def test_lowy_ingests_article_template_with_human_date_text(
    tmp_path: Path,
) -> None:
    """Accept Lowy pages that expose date text without datetime attribute."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    entries = [
        DiscoveredSitemapEntry(
            source_url="https://www.lowyinstitute.org/publications/human-date",
            published_at="2026-01-01T00:00:00+00:00",
        )
    ]

    html_bytes = (
        b"<html><article><h1>Title</h1><time>19 August 2025</time><p>"
        + (b"Human date article text. " * 120)
        + b"</p></article></html>"
    )

    def fetch_document(url: str) -> tuple[str, bytes]:
        assert url.endswith("/human-date")
        return "text/html", html_bytes

    report = ingest_source_documents(
        source_id="lowy_institute",
        sitemap_xml="",
        sqlite_path=sqlite_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
        discovered_entries=entries,
    )

    assert report.ingested_documents == 1


def test_lowy_page_date_overrides_listing_hint_for_published_at(
    tmp_path: Path,
) -> None:
    """Persist article-page publication date as canonical published_at."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    entries = [
        DiscoveredSitemapEntry(
            source_url="https://www.lowyinstitute.org/publications/date-override",
            published_at="2024-01-01T00:00:00+00:00",
        )
    ]

    html_bytes = (
        b"<html><main><time datetime='2025-08-19T00:00:00+00:00'></time><p>"
        + (b"Canonical date text. " * 120)
        + b"</p></main></html>"
    )

    def fetch_document(url: str) -> tuple[str, bytes]:
        assert url.endswith("/date-override")
        return "text/html", html_bytes

    report = ingest_source_documents(
        source_id="lowy_institute",
        sitemap_xml="",
        sqlite_path=sqlite_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
        discovered_entries=entries,
    )

    assert report.ingested_documents == 1

    versions = get_source_document_versions(
        sqlite_path=sqlite_path,
        source_id="lowy_institute",
    )
    assert len(versions) == 1
    assert versions[0].published_at == "2025-08-19T00:00:00+00:00"


def test_lowy_keeps_references_in_normalized_text(
    tmp_path: Path,
) -> None:
    """Preserve references section content in normalized text output."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    entries = [
        DiscoveredSitemapEntry(
            source_url="https://www.lowyinstitute.org/publications/references",
            published_at="2025-08-19T00:00:00+00:00",
        )
    ]

    html_bytes = (
        b"<html><main><time datetime='2025-08-19T00:00:00+00:00'></time>"
        b"<h1>Title</h1><p>"
        + (b"Main analysis body. " * 120)
        + b"</p><h2>References</h2><p>[1] Useful citation.</p>"
        b"</main></html>"
    )

    def fetch_document(url: str) -> tuple[str, bytes]:
        assert url.endswith("/references")
        return "text/html", html_bytes

    report = ingest_source_documents(
        source_id="lowy_institute",
        sitemap_xml="",
        sqlite_path=sqlite_path,
        artifact_dir=artifact_dir,
        fetch_document=fetch_document,
        discovered_entries=entries,
    )

    assert report.ingested_documents == 1

    versions = get_source_document_versions(
        sqlite_path=sqlite_path,
        source_id="lowy_institute",
    )
    assert len(versions) == 1
    assert "References" in versions[0].normalized_text
    assert "Useful citation." in versions[0].normalized_text
