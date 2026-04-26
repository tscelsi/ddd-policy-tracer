from __future__ import annotations

from io import StringIO
from pathlib import Path

from ddd_policy_tracer import get_source_document_versions
from ddd_policy_tracer.cli import run_cli


def test_cli_runs_manual_acquisition_for_source_and_prints_result(tmp_path: Path) -> None:
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
        fetch_document=lambda _url: ("text/plain", b"CLI ingestion content"),
        stdout=output,
    )

    assert exit_code == 0
    rendered = output.getvalue()
    assert "source=australia_institute" in rendered
    assert "run_status=completed" in rendered

    versions = get_source_document_versions(sqlite_path=sqlite_path, source_id="australia_institute")
    assert len(versions) == 1


def test_cli_limit_constrains_processing_scope(tmp_path: Path) -> None:
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
        fetch_document=lambda _url: ("text/plain", b"Limited run content"),
        stdout=output,
    )

    assert exit_code == 0
    assert "processed_urls=1" in output.getvalue()

    versions = get_source_document_versions(sqlite_path=sqlite_path, source_id="australia_institute")
    assert len(versions) == 1


def test_cli_dry_run_reports_discovery_without_persisting_state(tmp_path: Path) -> None:
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

    versions = get_source_document_versions(sqlite_path=sqlite_path, source_id="australia_institute")
    assert versions == []
