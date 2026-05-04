"""Unit tests for AustraliaInstituteSourceStrategy.discover_documents.

All tests exercise the discovery method in isolation using a fake ``fetch``
callable; no real HTTP is performed.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ddd_policy_tracer.discovery.source_strategies import (
    AustraliaInstituteSourceStrategy,
)

# ---------------------------------------------------------------------------
# Sitemap XML fixtures
# ---------------------------------------------------------------------------

_SITEMAP_INDEX = """\
<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap>
    <loc>https://australiainstitute.org.au/tai_cpt_report-sitemap.xml</loc>
  </sitemap>
  <sitemap>
    <loc>https://australiainstitute.org.au/tai_cpt_post-sitemap.xml</loc>
  </sitemap>
</sitemapindex>
"""

_REPORT_SITEMAP = """\
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://australiainstitute.org.au/report/alpha/</loc>
    <lastmod>2024-06-15</lastmod>
  </url>
  <url>
    <loc>https://australiainstitute.org.au/report/beta/</loc>
    <lastmod>2024-03-10</lastmod>
  </url>
  <url>
    <loc>https://australiainstitute.org.au/report/gamma/</loc>
    <lastmod>2023-11-01</lastmod>
  </url>
</urlset>
"""

_REPORT_SITEMAP_NO_DATES = """\
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://australiainstitute.org.au/report/no-date-1/</loc>
  </url>
  <url>
    <loc>https://australiainstitute.org.au/report/no-date-2/</loc>
  </url>
</urlset>
"""

_EMPTY_URLSET = """\
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
</urlset>
"""

# ---------------------------------------------------------------------------
# Fake fetch helpers
# ---------------------------------------------------------------------------

_REPORT_SITEMAP_URL = (
    "https://australiainstitute.org.au/tai_cpt_report-sitemap.xml"
)
_INDEX_URL = "https://australiainstitute.org.au/sitemap_index.xml"


def _make_fetch(responses: dict[str, str]):
    """Return a fetch callable that serves pre-defined URL -> XML responses."""

    def fetch(url: str, _user_agent: str) -> str:
        if url not in responses:
            raise KeyError(f"Unexpected fetch for URL: {url}")
        return responses[url]

    return fetch


def _default_fetch():
    """Return a fetch with the standard index + one report sitemap."""
    return _make_fetch(
        {
            _INDEX_URL: _SITEMAP_INDEX,
            _REPORT_SITEMAP_URL: _REPORT_SITEMAP,
        },
    )


# ---------------------------------------------------------------------------
# Strategy under test
# ---------------------------------------------------------------------------

STRATEGY = AustraliaInstituteSourceStrategy()


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_returns_all_entries_when_no_filters():
    """All sitemap entries are returned when no date filter or limit is set."""
    entries, selected_sitemaps = STRATEGY.discover_documents(
        fetch=_default_fetch(),
        published_since=None,
        limit=None,
    )

    urls = {e.source_url for e in entries}
    assert urls == {
        "https://australiainstitute.org.au/report/alpha/",
        "https://australiainstitute.org.au/report/beta/",
        "https://australiainstitute.org.au/report/gamma/",
    }


def test_selected_sitemaps_count_matches_matched_children():
    """The second return value equals the number of matched child sitemaps."""
    _, selected_sitemaps = STRATEGY.discover_documents(
        fetch=_default_fetch(),
        published_since=None,
        limit=None,
    )
    # Only one child matches "tai_cpt_report-sitemap"
    assert selected_sitemaps == 1


# ---------------------------------------------------------------------------
# published_since filter tests
# ---------------------------------------------------------------------------


def test_published_since_excludes_entries_before_cutoff():
    """Entries whose lastmod is earlier than published_since are excluded."""
    cutoff = datetime(2024, 4, 1, tzinfo=UTC)

    entries, _ = STRATEGY.discover_documents(
        fetch=_default_fetch(),
        published_since=cutoff,
        limit=None,
    )

    urls = {e.source_url for e in entries}
    assert urls == {"https://australiainstitute.org.au/report/alpha/"}


def test_published_since_on_exact_date_is_inclusive():
    """An entry whose lastmod equals published_since is included."""
    # beta's lastmod is 2024-03-10
    cutoff = datetime(2024, 3, 10, tzinfo=UTC)

    entries, _ = STRATEGY.discover_documents(
        fetch=_default_fetch(),
        published_since=cutoff,
        limit=None,
    )

    urls = {e.source_url for e in entries}
    assert "https://australiainstitute.org.au/report/beta/" in urls


def test_published_since_excludes_entries_with_no_date():
    """Entries that have no lastmod are dropped when published_since is set."""
    fetch = _make_fetch(
        {
            _INDEX_URL: _SITEMAP_INDEX,
            _REPORT_SITEMAP_URL: _REPORT_SITEMAP_NO_DATES,
        },
    )
    cutoff = datetime(2020, 1, 1, tzinfo=UTC)

    entries, _ = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=cutoff,
        limit=None,
    )

    assert entries == []


def test_published_since_none_includes_entries_with_no_date():
    """Entries without a publication date are included when filter is absent."""
    fetch = _make_fetch(
        {
            _INDEX_URL: _SITEMAP_INDEX,
            _REPORT_SITEMAP_URL: _REPORT_SITEMAP_NO_DATES,
        },
    )

    entries, _ = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=None,
        limit=None,
    )

    assert len(entries) == 2


def test_published_since_returns_empty_when_all_entries_predate_cutoff():
    """An empty list is returned when every entry predates the cutoff."""
    far_future = datetime(2099, 1, 1, tzinfo=UTC)

    entries, _ = STRATEGY.discover_documents(
        fetch=_default_fetch(),
        published_since=far_future,
        limit=None,
    )

    assert entries == []


# ---------------------------------------------------------------------------
# limit tests
# ---------------------------------------------------------------------------


def test_limit_truncates_result_list():
    """Only `limit` entries are returned when more are available."""
    entries, _ = STRATEGY.discover_documents(
        fetch=_default_fetch(),
        published_since=None,
        limit=2,
    )

    assert len(entries) == 2


def test_limit_one_returns_single_entry():
    """limit=1 yields exactly one entry."""
    entries, _ = STRATEGY.discover_documents(
        fetch=_default_fetch(),
        published_since=None,
        limit=1,
    )

    assert len(entries) == 1


def test_limit_zero_returns_empty_list():
    """limit=0 yields an empty list."""
    entries, _ = STRATEGY.discover_documents(
        fetch=_default_fetch(),
        published_since=None,
        limit=0,
    )

    assert entries == []


def test_limit_larger_than_available_returns_all():
    """A limit larger than the total result count returns all entries."""
    entries, _ = STRATEGY.discover_documents(
        fetch=_default_fetch(),
        published_since=None,
        limit=9999,
    )

    assert len(entries) == 3


def test_limit_none_returns_all_entries():
    """limit=None applies no restriction on the result length."""
    entries, _ = STRATEGY.discover_documents(
        fetch=_default_fetch(),
        published_since=None,
        limit=None,
    )

    assert len(entries) == 3


# ---------------------------------------------------------------------------
# Edge-cases: empty / no-matching-children
# ---------------------------------------------------------------------------


def test_no_matching_child_sitemaps_returns_empty_list():
    """No entries are returned when no child sitemaps match the pattern."""
    sitemap_index_no_reports = """\
<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap>
    <loc>https://australiainstitute.org.au/tai_cpt_post-sitemap.xml</loc>
  </sitemap>
</sitemapindex>
"""
    fetch = _make_fetch({_INDEX_URL: sitemap_index_no_reports})

    entries, selected_sitemaps = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=None,
        limit=None,
    )

    assert entries == []
    assert selected_sitemaps == 0


def test_empty_child_sitemap_returns_empty_list():
    """No entries are returned when the matching child sitemap is empty."""
    fetch = _make_fetch(
        {
            _INDEX_URL: _SITEMAP_INDEX,
            _REPORT_SITEMAP_URL: _EMPTY_URLSET,
        },
    )

    entries, _ = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=None,
        limit=None,
    )

    assert entries == []


def test_root_xml_is_urlset_not_index():
    """Handles sources that serve a plain urlset rather than a sitemap index."""
    # When the root URL itself returns a urlset, it is used directly.
    fetch = _make_fetch({_INDEX_URL: _REPORT_SITEMAP})

    entries, selected_sitemaps = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=None,
        limit=None,
    )

    assert len(entries) == 3
    assert selected_sitemaps == 1


def test_discovered_entries_carry_correct_published_at():
    """Each returned entry's published_at matches its sitemap lastmod value."""
    entries, _ = STRATEGY.discover_documents(
        fetch=_default_fetch(),
        published_since=None,
        limit=None,
    )

    by_url = {e.source_url: e.published_at for e in entries}
    assert (
        by_url["https://australiainstitute.org.au/report/alpha/"]
        == "2024-06-15"
    )
    assert (
        by_url["https://australiainstitute.org.au/report/beta/"] == "2024-03-10"
    )
    assert (
        by_url["https://australiainstitute.org.au/report/gamma/"]
        == "2023-11-01"
    )


def test_duplicate_urls_in_sitemap_are_deduplicated():
    """Duplicate URLs in a sitemap produce only one entry."""
    sitemap_with_dupes = """\
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://australiainstitute.org.au/report/dup/</loc>
    <lastmod>2024-01-01</lastmod>
  </url>
  <url>
    <loc>https://australiainstitute.org.au/report/dup/</loc>
    <lastmod>2024-06-01</lastmod>
  </url>
</urlset>
"""
    fetch = _make_fetch(
        {
            _INDEX_URL: _SITEMAP_INDEX,
            _REPORT_SITEMAP_URL: sitemap_with_dupes,
        },
    )

    entries, _ = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=None,
        limit=None,
    )

    urls = [e.source_url for e in entries]
    assert urls.count("https://australiainstitute.org.au/report/dup/") == 1


def test_combined_published_since_and_limit():
    """published_since is applied before limit, so limit counts filtered results."""
    # After filtering to entries >= 2024-04-01 there is only one (alpha).
    # A limit of 5 should still return just that one.
    cutoff = datetime(2024, 4, 1, tzinfo=UTC)

    entries, _ = STRATEGY.discover_documents(
        fetch=_default_fetch(),
        published_since=cutoff,
        limit=5,
    )

    assert len(entries) == 1
    assert (
        entries[0].source_url
        == "https://australiainstitute.org.au/report/alpha/"
    )
