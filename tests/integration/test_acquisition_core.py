from pathlib import Path

from ddd_policy_tracer import get_source_document_versions, ingest_source_documents


def test_ingest_from_sitemap_persists_first_source_document_version(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://australiainstitute.org.au/report-1?utm_source=newsletter</loc></url>
    </urlset>
    """.strip()

    def fetch_document(url: str) -> tuple[str, bytes]:
        assert "report-1" in url
        return "text/plain", b"This is a report about policy reform."

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

    versions = get_source_document_versions(
        sqlite_path=sqlite_path,
        source_id="australia_institute",
    )

    assert len(versions) == 1
    version = versions[0]
    assert version.source_document_id == "https://australiainstitute.org.au/report-1"
    assert version.normalized_text == "This is a report about policy reform."
    assert version.raw_content_ref.startswith(str(artifact_dir))
    assert Path(version.raw_content_ref).exists()
