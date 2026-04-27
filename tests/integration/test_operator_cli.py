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
      <url><loc>https://australiainstitute.org.au/report-1</loc></url>
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
