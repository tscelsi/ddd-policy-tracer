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
    assert report.run_status == "completed"

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


def test_reprocessing_same_unchanged_url_is_idempotent(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://australiainstitute.org.au/report-1?utm_source=newsletter</loc></url>
    </urlset>
    """.strip()

    def fetch_document(_: str) -> tuple[str, bytes]:
        return "text/plain", b"Unchanged report body."

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

    versions = get_source_document_versions(sqlite_path=sqlite_path, source_id="australia_institute")
    assert len(versions) == 1


def test_checksum_change_appends_new_version(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://australiainstitute.org.au/report-1</loc></url>
    </urlset>
    """.strip()

    payloads = [
        b"Version 1 content.",
        b"Version 2 content with updates.",
    ]

    def fetch_document(_: str) -> tuple[str, bytes]:
        return "text/plain", payloads.pop(0)

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

    versions = get_source_document_versions(sqlite_path=sqlite_path, source_id="australia_institute")
    assert len(versions) == 2
    assert versions[0].checksum != versions[1].checksum


def test_run_status_is_completed_with_failures_for_mixed_outcomes(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://australiainstitute.org.au/report-ok</loc></url>
      <url><loc>https://australiainstitute.org.au/report-fail</loc></url>
    </urlset>
    """.strip()

    def fetch_document(url: str) -> tuple[str, bytes]:
        if "report-fail" in url:
            raise RuntimeError("temporary fetch failure")
        return "text/plain", b"Healthy document content."

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


def test_compliance_blocks_disallowed_url_and_uses_configured_user_agent(tmp_path: Path) -> None:
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
        return "text/plain", b"Compliant content."

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

    assert fetched_urls == ["https://australiainstitute.org.au/allowed-report"]
    assert seen_user_agents == ["policy-tracer/0.1"]
    assert report.processed_urls == 2
    assert report.ingested_documents == 1
    assert report.failed_documents == 0
    assert report.skipped_urls == 1


def test_transient_failure_is_retried_and_observable(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://australiainstitute.org.au/retry-report</loc></url>
    </urlset>
    """.strip()

    attempts = 0

    def fetch_document(_: str) -> tuple[str, bytes]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("temporary timeout")
        return "text/plain", b"Recovered after retries."

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


def test_terminal_failure_records_actionable_reason_after_retry_budget_exhausted(
    tmp_path: Path,
) -> None:
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


def test_domain_events_are_emitted_in_expected_lifecycle_sequence(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "acquisition.db"
    artifact_dir = tmp_path / "artifacts"
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://australiainstitute.org.au/event-ok</loc></url>
      <url><loc>https://australiainstitute.org.au/event-fail</loc></url>
    </urlset>
    """.strip()

    def fetch_document(url: str) -> tuple[str, bytes]:
        if "event-fail" in url:
            raise RuntimeError("boom")
        return "text/plain", b"Eventful content"

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
    assert all(event.source_id == "australia_institute" for event in report.events)
    assert report.events[1].source_url == "https://australiainstitute.org.au/event-ok"
    assert report.events[2].source_url == "https://australiainstitute.org.au/event-fail"
    assert report.events[3].run_status == "completed_with_failures"
