"""Integration tests for operator CLI acquisition execution paths."""

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from ddd_policy_tracer import get_source_document_versions
from ddd_policy_tracer.cli import run_cli


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


def test_cli_runs_manual_acquisition_for_source_and_prints_result(
    tmp_path: Path,
) -> None:
    """Run CLI acquisition and verify persisted output and rendered summary."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_path = tmp_path / "sitemap.xml"
    sitemap_path.write_text(
        """
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://australiainstitute.org.au/report-1</loc></url>
        </urlset>
        """.strip(),
        encoding="utf-8",
    )

    output = StringIO()

    exit_code = run_cli(
        [
            "acquire",
            "--source",
            "australia_institute",
            "--sitemap-xml-path",
            str(sitemap_path),
            "--sqlite-path",
            str(sqlite_path),
            "--artifact-dir",
            str(artifact_dir),
        ],
        fetch_document=lambda url: (
            (
                "text/html",
                _report_html_with_pdf_link(
                    "https://australiainstitute.org.au/wp-content/report-1.pdf"
                ),
            )
            if url.endswith("/report-1")
            else (
                "application/pdf",
                _build_pdf_with_text("CLI ingestion content"),
            )
        ),
        stdout=output,
    )

    assert exit_code == 0
    rendered = output.getvalue()
    assert "source=australia_institute" in rendered
    assert "run_status=completed" in rendered

    versions = get_source_document_versions(
        sqlite_path=sqlite_path, source_id="australia_institute"
    )
    assert len(versions) == 1


def test_cli_limit_constrains_processing_scope(tmp_path: Path) -> None:
    """Ensure CLI limit flag caps processed sitemap URLs."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_path = tmp_path / "sitemap.xml"
    sitemap_path.write_text(
        """
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://australiainstitute.org.au/report-1</loc></url>
          <url><loc>https://australiainstitute.org.au/report-2</loc></url>
        </urlset>
        """.strip(),
        encoding="utf-8",
    )

    output = StringIO()

    exit_code = run_cli(
        [
            "acquire",
            "--source",
            "australia_institute",
            "--sitemap-xml-path",
            str(sitemap_path),
            "--sqlite-path",
            str(sqlite_path),
            "--artifact-dir",
            str(artifact_dir),
            "--limit",
            "1",
        ],
        fetch_document=lambda url: (
            (
                "text/html",
                _report_html_with_pdf_link(
                    "https://australiainstitute.org.au/wp-content/report.pdf"
                ),
            )
            if url.endswith("/report-1")
            else (
                "application/pdf",
                _build_pdf_with_text("Limited run content"),
            )
        ),
        stdout=output,
    )

    assert exit_code == 0
    assert "processed_urls=1" in output.getvalue()

    versions = get_source_document_versions(
        sqlite_path=sqlite_path, source_id="australia_institute"
    )
    assert len(versions) == 1


def test_cli_dry_run_reports_discovery_without_persisting_state(
    tmp_path: Path,
) -> None:
    """Ensure dry-run reports discovery without fetching or persistence."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_path = tmp_path / "sitemap.xml"
    sitemap_path.write_text(
        """
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://australiainstitute.org.au/report-1</loc></url>
          <url><loc>https://australiainstitute.org.au/report-2</loc></url>
        </urlset>
        """.strip(),
        encoding="utf-8",
    )

    output = StringIO()
    fetch_calls: list[str] = []

    def fetch_document(url: str) -> tuple[str, bytes]:
        fetch_calls.append(url)
        return "text/plain", b"Should not run in dry mode"

    exit_code = run_cli(
        [
            "acquire",
            "--source",
            "australia_institute",
            "--sitemap-xml-path",
            str(sitemap_path),
            "--sqlite-path",
            str(sqlite_path),
            "--artifact-dir",
            str(artifact_dir),
            "--dry-run",
            "--limit",
            "1",
        ],
        fetch_document=fetch_document,
        stdout=output,
    )

    assert exit_code == 0
    rendered = output.getvalue()
    assert "dry_run" in rendered
    assert "discovered_urls=1" in rendered
    assert fetch_calls == []
    assert not sqlite_path.exists()

    versions = get_source_document_versions(
        sqlite_path=sqlite_path, source_id="australia_institute"
    )
    assert versions == []


def test_cli_can_resolve_sitemap_index_using_child_pattern(
    tmp_path: Path,
) -> None:
    """Resolve sitemap index URLs and ingest matching child sitemap entries."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    output = StringIO()

    sitemap_index_xml = """
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>https://australiainstitute.org.au/tai_cpt_report-sitemap.xml</loc></sitemap>
      <sitemap><loc>https://australiainstitute.org.au/page-sitemap.xml</loc></sitemap>
    </sitemapindex>
    """.strip()
    child_report_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url>
        <loc>https://australiainstitute.org.au/report-1</loc>
        <lastmod>2026-04-20T09:30:00+00:00</lastmod>
      </url>
    </urlset>
    """.strip()
    child_page_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://australiainstitute.org.au/about</loc></url>
    </urlset>
    """.strip()

    def fetch_text_url(url: str, _user_agent: str) -> str:
        if url.endswith("sitemap_index.xml"):
            return sitemap_index_xml
        if url.endswith("tai_cpt_report-sitemap.xml"):
            return child_report_xml
        return child_page_xml

    exit_code = run_cli(
        [
            "acquire",
            "--source",
            "australia_institute",
            "--sitemap-url",
            "https://australiainstitute.org.au/sitemap_index.xml",
            "--child-sitemap-pattern",
            "tai_cpt_report-sitemap",
            "--sqlite-path",
            str(sqlite_path),
            "--artifact-dir",
            str(artifact_dir),
        ],
        fetch_document=lambda url, _ua: (
            (
                "text/html",
                _report_html_with_pdf_link(
                    "https://australiainstitute.org.au/wp-content/index.pdf"
                ),
            )
            if url.endswith("/report-1")
            else ("application/pdf", _build_pdf_with_text("index-based run"))
        ),
        fetch_text_url=fetch_text_url,
        stdout=output,
    )

    assert exit_code == 0
    assert "processed_urls=1" in output.getvalue()

    versions = get_source_document_versions(
        sqlite_path=sqlite_path, source_id="australia_institute"
    )
    assert len(versions) == 1
    assert versions[0].published_at == "2026-04-20T09:30:00+00:00"


def test_cli_supports_filesystem_repository_backend(tmp_path: Path) -> None:
    """Persist CLI ingestion state using filesystem repository backend."""
    state_path = tmp_path / "acquisition.jsonl"
    artifact_dir = tmp_path / "artifacts"
    sitemap_path = tmp_path / "sitemap.xml"
    sitemap_path.write_text(
        """
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://australiainstitute.org.au/report-1</loc></url>
        </urlset>
        """.strip(),
        encoding="utf-8",
    )

    output = StringIO()

    exit_code = run_cli(
        [
            "acquire",
            "--source",
            "australia_institute",
            "--sitemap-xml-path",
            str(sitemap_path),
            "--sqlite-path",
            str(state_path),
            "--repository-backend",
            "filesystem",
            "--artifact-dir",
            str(artifact_dir),
        ],
        fetch_document=lambda url: (
            (
                "text/html",
                _report_html_with_pdf_link(
                    "https://australiainstitute.org.au/wp-content/report-1.pdf"
                ),
            )
            if url.endswith("/report-1")
            else (
                "application/pdf",
                _build_pdf_with_text("CLI filesystem backend content"),
            )
        ),
        stdout=output,
    )

    assert exit_code == 0
    assert "run_status=completed" in output.getvalue()

    versions = get_source_document_versions(
        sqlite_path=state_path,
        source_id="australia_institute",
        repository_backend="filesystem",
    )
    assert len(versions) == 1


def test_cli_published_within_years_filters_old_entries(
    tmp_path: Path,
) -> None:
    """Process only sitemap entries within the configured year window."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_path = tmp_path / "sitemap.xml"
    sitemap_path.write_text(
        """
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url>
            <loc>https://australiainstitute.org.au/report-recent</loc>
            <lastmod>2026-01-01T00:00:00+00:00</lastmod>
          </url>
          <url>
            <loc>https://australiainstitute.org.au/report-old</loc>
            <lastmod>2019-01-01T00:00:00+00:00</lastmod>
          </url>
        </urlset>
        """.strip(),
        encoding="utf-8",
    )

    output = StringIO()

    exit_code = run_cli(
        [
            "acquire",
            "--source",
            "australia_institute",
            "--sitemap-xml-path",
            str(sitemap_path),
            "--sqlite-path",
            str(sqlite_path),
            "--artifact-dir",
            str(artifact_dir),
            "--published-within-years",
            "2",
        ],
        fetch_document=lambda url: (
            (
                "text/html",
                _report_html_with_pdf_link(
                    "https://australiainstitute.org.au/wp-content/recent.pdf"
                ),
            )
            if url.endswith("/report-recent")
            else (
                "application/pdf",
                _build_pdf_with_text("Recent CLI content"),
            )
        ),
        stdout=output,
    )

    assert exit_code == 0
    assert "processed_urls=2" in output.getvalue()
    assert "skipped_urls=1" in output.getvalue()

    versions = get_source_document_versions(
        sqlite_path=sqlite_path,
        source_id="australia_institute",
    )
    assert len(versions) == 1
    assert versions[0].source_url.endswith("/report-recent")


def test_cli_dry_run_includes_published_since_details(
    tmp_path: Path,
) -> None:
    """Render publish-time filter details in dry-run output."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_path = tmp_path / "sitemap.xml"
    sitemap_path.write_text(
        """
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url>
            <loc>https://australiainstitute.org.au/report-1</loc>
            <lastmod>2026-01-01T00:00:00+00:00</lastmod>
          </url>
        </urlset>
        """.strip(),
        encoding="utf-8",
    )

    output = StringIO()

    exit_code = run_cli(
        [
            "acquire",
            "--source",
            "australia_institute",
            "--sitemap-xml-path",
            str(sitemap_path),
            "--sqlite-path",
            str(sqlite_path),
            "--artifact-dir",
            str(artifact_dir),
            "--dry-run",
            "--published-since",
            "2025-01-01T00:00:00+00:00",
        ],
        fetch_document=lambda _url: ("text/plain", b"unused"),
        stdout=output,
    )

    assert exit_code == 0
    rendered = output.getvalue()
    assert "dry_run" in rendered
    assert "published_since=2025-01-01T00:00:00+00:00" in rendered


def test_cli_lowy_requires_limit_or_publish_filter(
    tmp_path: Path,
) -> None:
    """Require Lowy runs to include limit or published-time bounds."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    output = StringIO()

    try:
        run_cli(
            [
                "acquire",
                "--source",
                "lowy_institute",
                "--sqlite-path",
                str(sqlite_path),
                "--artifact-dir",
                str(artifact_dir),
            ],
            fetch_document=lambda _url: ("text/plain", b"unused"),
            fetch_text_url=lambda _url, _ua: "",
            stdout=output,
        )
    except ValueError as exc:
        assert "requires either --limit or --published" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unbounded Lowy run")


def test_cli_lowy_dry_run_discovers_publication_urls_with_limit(
    tmp_path: Path,
) -> None:
    """Discover Lowy listing URLs in dry-run and honor limit."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    output = StringIO()

    listing_page = """
    <html><body>
      <article>
        <a href="/publications/first-report">First</a>
        <time datetime="2026-01-02T00:00:00+00:00">2 Jan 2026</time>
      </article>
      <article>
        <a href="/publications/second-report">Second</a>
        <time datetime="2026-01-01T00:00:00+00:00">1 Jan 2026</time>
      </article>
      <article>
        <a href="/publications/third-report">Third</a>
        <time datetime="2025-12-31T00:00:00+00:00">31 Dec 2025</time>
      </article>
    </body></html>
    """.strip()

    def fetch_text_url(url: str, _user_agent: str) -> str:
        if url.endswith("?page=0"):
            return listing_page
        return "<html><body></body></html>"

    exit_code = run_cli(
        [
            "acquire",
            "--source",
            "lowy_institute",
            "--sqlite-path",
            str(sqlite_path),
            "--artifact-dir",
            str(artifact_dir),
            "--limit",
            "2",
            "--dry-run",
        ],
        fetch_document=lambda _url: ("text/plain", b"unused"),
        fetch_text_url=fetch_text_url,
        stdout=output,
    )

    assert exit_code == 0
    rendered = output.getvalue()
    assert "dry_run" in rendered
    assert "source=lowy_institute" in rendered
    assert "discovered_urls=2" in rendered


def test_cli_lowy_dry_run_stops_after_first_older_dated_listing_item(
    tmp_path: Path,
) -> None:
    """Stop Lowy pagination after crossing date cutoff."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    output = StringIO()
    fetched_urls: list[str] = []

    listing_page_zero = """
    <html><body>
      <article>
        <a href="/publications/new-report">New</a>
        <time datetime="2025-06-01T00:00:00+00:00">1 June 2025</time>
      </article>
      <article>
        <a href="/publications/old-report">Old</a>
        <time datetime="2024-01-01T00:00:00+00:00">1 Jan 2024</time>
      </article>
    </body></html>
    """.strip()

    def fetch_text_url(url: str, _user_agent: str) -> str:
        fetched_urls.append(url)
        if url.endswith("?page=0"):
            return listing_page_zero
        return "<html><body></body></html>"

    exit_code = run_cli(
        [
            "acquire",
            "--source",
            "lowy_institute",
            "--sqlite-path",
            str(sqlite_path),
            "--artifact-dir",
            str(artifact_dir),
            "--published-since",
            "2025-01-01T00:00:00+00:00",
            "--dry-run",
        ],
        fetch_document=lambda _url: ("text/plain", b"unused"),
        fetch_text_url=fetch_text_url,
        stdout=output,
    )

    assert exit_code == 0
    rendered = output.getvalue()
    assert "discovered_urls=1" in rendered
    assert fetched_urls == ["https://www.lowyinstitute.org/publications?page=0"]


def test_cli_lowy_dry_run_continues_when_listing_item_is_undated(
    tmp_path: Path,
) -> None:
    """Do not stop pagination early when boundary listing item is undated."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    output = StringIO()
    fetched_urls: list[str] = []

    listing_page_zero = """
    <html><body>
      <article>
        <a href="/publications/new-report">New</a>
        <time datetime="2025-06-01T00:00:00+00:00">1 June 2025</time>
      </article>
      <article>
        <a href="/publications/undated-report">Undated</a>
      </article>
    </body></html>
    """.strip()

    listing_page_one = """
    <html><body>
      <article>
        <a href="/publications/old-report">Old</a>
        <time datetime="2024-01-01T00:00:00+00:00">1 Jan 2024</time>
      </article>
    </body></html>
    """.strip()

    def fetch_text_url(url: str, _user_agent: str) -> str:
        fetched_urls.append(url)
        if url.endswith("?page=0"):
            return listing_page_zero
        if url.endswith("?page=1"):
            return listing_page_one
        return "<html><body></body></html>"

    exit_code = run_cli(
        [
            "acquire",
            "--source",
            "lowy_institute",
            "--sqlite-path",
            str(sqlite_path),
            "--artifact-dir",
            str(artifact_dir),
            "--published-since",
            "2025-01-01T00:00:00+00:00",
            "--dry-run",
        ],
        fetch_document=lambda _url: ("text/plain", b"unused"),
        fetch_text_url=fetch_text_url,
        stdout=output,
    )

    assert exit_code == 0
    rendered = output.getvalue()
    assert "discovered_urls=2" in rendered
    assert fetched_urls == [
        "https://www.lowyinstitute.org/publications?page=0",
        "https://www.lowyinstitute.org/publications?page=1",
    ]


def test_cli_lowy_dry_run_uses_published_within_years_bound(
    tmp_path: Path,
) -> None:
    """Apply published-within-years cutoff to Lowy listing discovery."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    output = StringIO()

    listing_page_zero = """
    <html><body>
      <article>
        <a href="/publications/recent-report">Recent</a>
        <time datetime="2026-01-01T00:00:00+00:00">1 Jan 2026</time>
      </article>
      <article>
        <a href="/publications/old-report">Old</a>
        <time datetime="2020-01-01T00:00:00+00:00">1 Jan 2020</time>
      </article>
    </body></html>
    """.strip()

    def fetch_text_url(url: str, _user_agent: str) -> str:
        if url.endswith("?page=0"):
            return listing_page_zero
        return "<html><body></body></html>"

    exit_code = run_cli(
        [
            "acquire",
            "--source",
            "lowy_institute",
            "--sqlite-path",
            str(sqlite_path),
            "--artifact-dir",
            str(artifact_dir),
            "--published-within-years",
            "2",
            "--dry-run",
        ],
        fetch_document=lambda _url: ("text/plain", b"unused"),
        fetch_text_url=fetch_text_url,
        stdout=output,
    )

    assert exit_code == 0
    rendered = output.getvalue()
    assert "source=lowy_institute" in rendered
    assert "discovered_urls=1" in rendered
    assert "published_since=" in rendered


def test_cli_lowy_non_dry_run_uses_listing_discovery(
    tmp_path: Path,
) -> None:
    """Run non-dry Lowy acquisition without sitemap arguments."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    output = StringIO()

    listing_page_zero = """
    <html><body>
      <article>
        <a href="/publications/first-report">First</a>
        <time datetime="2026-01-01T00:00:00+00:00">1 Jan 2026</time>
      </article>
      <article>
        <a href="/publications/second-report">Second</a>
        <time datetime="2025-12-30T00:00:00+00:00">30 Dec 2025</time>
      </article>
    </body></html>
    """.strip()

    def fetch_text_url(url: str, _user_agent: str) -> str:
        if url.endswith("?page=0"):
            return listing_page_zero
        return "<html><body></body></html>"

    exit_code = run_cli(
        [
            "acquire",
            "--source",
            "lowy_institute",
            "--sqlite-path",
            str(sqlite_path),
            "--artifact-dir",
            str(artifact_dir),
            "--limit",
            "2",
        ],
        fetch_document=lambda _url: ("text/plain", b"unused"),
        fetch_text_url=fetch_text_url,
        stdout=output,
    )

    assert exit_code == 0
    rendered = output.getvalue()
    assert "source=lowy_institute" in rendered
    assert "processed_urls=2" in rendered
    assert "failed_documents=0" in rendered
    assert "skipped_urls=2" in rendered


def test_cli_lowy_non_dry_run_stops_on_older_dated_listing_item(
    tmp_path: Path,
) -> None:
    """Bound Lowy non-dry discovery when dated listing crosses cutoff."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    output = StringIO()

    listing_page_zero = """
    <html><body>
      <article>
        <a href="/publications/new-report">New</a>
        <time datetime="2025-06-01T00:00:00+00:00">1 June 2025</time>
      </article>
      <article>
        <a href="/publications/old-report">Old</a>
        <time datetime="2024-01-01T00:00:00+00:00">1 Jan 2024</time>
      </article>
    </body></html>
    """.strip()

    def fetch_text_url(url: str, _user_agent: str) -> str:
        if url.endswith("?page=0"):
            return listing_page_zero
        return "<html><body></body></html>"

    exit_code = run_cli(
        [
            "acquire",
            "--source",
            "lowy_institute",
            "--sqlite-path",
            str(sqlite_path),
            "--artifact-dir",
            str(artifact_dir),
            "--published-since",
            "2025-01-01T00:00:00+00:00",
        ],
        fetch_document=lambda _url: ("text/plain", b"unused"),
        fetch_text_url=fetch_text_url,
        stdout=output,
    )

    assert exit_code == 0
    rendered = output.getvalue()
    assert "source=lowy_institute" in rendered
    assert "processed_urls=1" in rendered
    assert "failed_documents=0" in rendered
    assert "skipped_urls=1" in rendered


def test_cli_lowy_non_dry_run_continues_when_listing_item_is_undated(
    tmp_path: Path,
) -> None:
    """Continue Lowy non-dry pagination when boundary item lacks date."""
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    output = StringIO()
    fetched_urls: list[str] = []

    listing_page_zero = """
    <html><body>
      <article>
        <a href="/publications/new-report">New</a>
        <time datetime="2025-06-01T00:00:00+00:00">1 June 2025</time>
      </article>
      <article>
        <a href="/publications/undated-report">Undated</a>
      </article>
    </body></html>
    """.strip()

    listing_page_one = """
    <html><body>
      <article>
        <a href="/publications/old-report">Old</a>
        <time datetime="2024-01-01T00:00:00+00:00">1 Jan 2024</time>
      </article>
    </body></html>
    """.strip()

    def fetch_text_url(url: str, _user_agent: str) -> str:
        fetched_urls.append(url)
        if url.endswith("?page=0"):
            return listing_page_zero
        if url.endswith("?page=1"):
            return listing_page_one
        return "<html><body></body></html>"

    exit_code = run_cli(
        [
            "acquire",
            "--source",
            "lowy_institute",
            "--sqlite-path",
            str(sqlite_path),
            "--artifact-dir",
            str(artifact_dir),
            "--published-since",
            "2025-01-01T00:00:00+00:00",
        ],
        fetch_document=lambda _url: ("text/plain", b"unused"),
        fetch_text_url=fetch_text_url,
        stdout=output,
    )

    assert exit_code == 0
    rendered = output.getvalue()
    assert "source=lowy_institute" in rendered
    assert "processed_urls=2" in rendered
    assert "failed_documents=0" in rendered
    assert "skipped_urls=2" in rendered
    assert fetched_urls == [
        "https://www.lowyinstitute.org/publications?page=0",
        "https://www.lowyinstitute.org/publications?page=1",
    ]
