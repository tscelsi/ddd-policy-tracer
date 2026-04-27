"""Unit tests for sitemap discovery adapter behavior."""

from ddd_policy_tracer.adapters import discover_sitemap_entries


def test_discover_sitemap_entries_reads_loc_and_lastmod() -> None:
    """Parse sitemap entries and map lastmod to published_at."""
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url>
        <loc>https://australiainstitute.org.au/report/alpha/</loc>
        <lastmod>2026-04-20T09:30:00+00:00</lastmod>
      </url>
    </urlset>
    """.strip()

    entries = discover_sitemap_entries(sitemap_xml)

    assert len(entries) == 1
    assert (
        entries[0].source_url
        == "https://australiainstitute.org.au/report/alpha/"
    )
    assert entries[0].published_at == "2026-04-20T09:30:00+00:00"


def test_discover_sitemap_entries_allows_missing_or_invalid_lastmod() -> None:
    """Keep URLs discoverable when lastmod is missing or malformed."""
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://australiainstitute.org.au/report/no-lastmod/</loc></url>
      <url>
        <loc>https://australiainstitute.org.au/report/bad-lastmod/</loc>
        <lastmod>not-a-timestamp</lastmod>
      </url>
    </urlset>
    """.strip()

    entries = discover_sitemap_entries(sitemap_xml)

    assert [entry.source_url for entry in entries] == [
        "https://australiainstitute.org.au/report/no-lastmod/",
        "https://australiainstitute.org.au/report/bad-lastmod/",
    ]
    assert all(entry.published_at is None for entry in entries)


def test_discover_sitemap_entries_deduplicates_and_keeps_latest_lastmod(
) -> None:
    """Keep one entry per URL and retain the newest published_at value."""
    sitemap_xml = """
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url>
        <loc>https://australiainstitute.org.au/report/dupe/</loc>
        <lastmod>2026-04-20T09:30:00+00:00</lastmod>
      </url>
      <url>
        <loc>https://australiainstitute.org.au/report/dupe/</loc>
        <lastmod>2026-04-25T10:45:00+00:00</lastmod>
      </url>
    </urlset>
    """.strip()

    entries = discover_sitemap_entries(sitemap_xml)

    assert len(entries) == 1
    assert (
        entries[0].source_url
        == "https://australiainstitute.org.au/report/dupe/"
    )
    assert entries[0].published_at == "2026-04-25T10:45:00+00:00"
