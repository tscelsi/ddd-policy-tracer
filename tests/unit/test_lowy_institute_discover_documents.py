"""Unit tests for LowyInstituteSourceStrategy.discover_documents.

All tests exercise the discovery method in isolation using a fake ``fetch``
callable that serves pre-built HTML; no real HTTP is performed.

The Lowy strategy delegates entirely to ``discover_lowy_listing_entries``,
which paginates over listing pages, early-exits when an entry predates
``published_since``, and re-maps ``max_documents`` to ``limit``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ddd_policy_tracer.discovery.source_strategies import (
    LowyInstituteSourceStrategy,
)

# ---------------------------------------------------------------------------
# HTML listing-page builder helpers
# ---------------------------------------------------------------------------

_BASE_URL = "https://www.lowyinstitute.org/publications"


def _article_html(slug: str, date_iso: str | None = None) -> str:
    """Build one minimal Lowy publication article card."""
    time_tag = (
        f'<time datetime="{date_iso}">{date_iso}</time>'
        if date_iso is not None
        else ""
    )
    return (
        f'<article><a href="/publications/{slug}">Title</a>{time_tag}</article>'
    )


def _listing_page(*articles: str) -> str:
    """Wrap article HTML fragments in a minimal listing-page document."""
    body = "\n".join(articles)
    return f"<html><body>{body}</body></html>"


def _empty_page() -> str:
    """Return a listing page with no article cards (signals end of pages)."""
    return "<html><body></body></html>"


# ---------------------------------------------------------------------------
# Fake fetch helper
# ---------------------------------------------------------------------------


def _make_fetch(pages: dict[int, str]) -> object:
    """Return a fetch callable serving page-indexed HTML responses.

    Unrecognised page numbers return an empty page (pagination sentinel).
    """

    def fetch(url: str, _user_agent: str) -> str:
        # URL format: .../publications?page=N
        page_num = int(url.split("page=")[1]) if "page=" in url else 0
        return pages.get(page_num, _empty_page())

    return fetch


def _single_page_fetch(*articles: str):
    """Return a fetch that serves one page of articles then stops."""
    return _make_fetch({0: _listing_page(*articles)})


# ---------------------------------------------------------------------------
# Strategy under test
# ---------------------------------------------------------------------------

STRATEGY = LowyInstituteSourceStrategy()


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_returns_all_entries_from_single_page():
    """All entries on a single listing page are returned."""
    fetch = _single_page_fetch(
        _article_html("alpha", "2024-06-01T00:00:00+00:00"),
        _article_html("beta", "2024-05-01T00:00:00+00:00"),
    )

    entries, pages_scanned = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=None,
        limit=None,
    )

    urls = {e.source_url for e in entries}
    assert urls == {
        "https://www.lowyinstitute.org/publications/alpha",
        "https://www.lowyinstitute.org/publications/beta",
    }


def test_pages_scanned_equals_number_of_non_empty_pages_plus_one():
    """pages_scanned includes the empty sentinel page that terminated crawl."""
    fetch = _single_page_fetch(
        _article_html("alpha", "2024-06-01T00:00:00+00:00"),
    )

    _, pages_scanned = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=None,
        limit=None,
    )

    # page 0 has entries, page 1 is empty → 2 pages scanned
    assert pages_scanned == 2


def test_entries_carry_correct_published_at():
    """Each returned entry's published_at is preserved from the HTML datetime attr."""
    fetch = _single_page_fetch(
        _article_html("alpha", "2024-06-01T00:00:00+00:00"),
    )

    entries, _ = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=None,
        limit=None,
    )

    assert len(entries) == 1
    assert entries[0].published_at is not None
    assert "2024-06-01" in entries[0].published_at


def test_pagination_collects_entries_across_multiple_pages():
    """Entries from multiple listing pages are combined into one list."""
    fetch = _make_fetch(
        {
            0: _listing_page(
                _article_html("alpha", "2024-06-01T00:00:00+00:00"),
            ),
            1: _listing_page(
                _article_html("beta", "2024-05-01T00:00:00+00:00"),
            ),
        },
    )

    entries, _ = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=None,
        limit=None,
    )

    urls = {e.source_url for e in entries}
    assert "https://www.lowyinstitute.org/publications/alpha" in urls
    assert "https://www.lowyinstitute.org/publications/beta" in urls


# ---------------------------------------------------------------------------
# published_since filter tests
# ---------------------------------------------------------------------------


def test_published_since_stops_at_older_entry():
    """Discovery halts and returns only entries on or after published_since."""
    fetch = _single_page_fetch(
        _article_html("new", "2024-06-01T00:00:00+00:00"),
        _article_html("old", "2023-01-01T00:00:00+00:00"),
    )
    cutoff = datetime(2024, 1, 1, tzinfo=UTC)

    entries, _ = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=cutoff,
        limit=None,
    )

    urls = {e.source_url for e in entries}
    assert "https://www.lowyinstitute.org/publications/new" in urls
    assert "https://www.lowyinstitute.org/publications/old" not in urls


def test_published_since_returns_empty_when_first_entry_is_too_old():
    """Empty list returned when every visible entry predates the cutoff."""
    fetch = _single_page_fetch(
        _article_html("old", "2020-01-01T00:00:00+00:00"),
    )
    cutoff = datetime(2024, 1, 1, tzinfo=UTC)

    entries, _ = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=cutoff,
        limit=None,
    )

    assert entries == []


def test_published_since_includes_entries_with_matching_date():
    """An entry whose date exactly matches published_since is included."""
    fetch = _single_page_fetch(
        _article_html("exact", "2024-03-10T00:00:00+00:00"),
    )
    cutoff = datetime(2024, 3, 10, tzinfo=UTC)

    entries, _ = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=cutoff,
        limit=None,
    )

    assert len(entries) == 1


def test_published_since_skips_entries_without_dates_does_not_halt():
    """Dateless entries are included and do not trigger early exit."""
    fetch = _single_page_fetch(
        _article_html("no-date"),
        _article_html("dated", "2024-06-01T00:00:00+00:00"),
    )
    cutoff = datetime(2024, 1, 1, tzinfo=UTC)

    entries, _ = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=cutoff,
        limit=None,
    )

    urls = {e.source_url for e in entries}
    # dateless entry is appended; dated entry is also within range
    assert "https://www.lowyinstitute.org/publications/no-date" in urls
    assert "https://www.lowyinstitute.org/publications/dated" in urls


def test_published_since_far_future_returns_empty():
    """No entries returned when published_since is set far into the future."""
    fetch = _single_page_fetch(
        _article_html("alpha", "2024-06-01T00:00:00+00:00"),
    )
    far_future = datetime(2099, 1, 1, tzinfo=UTC)

    entries, _ = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=far_future,
        limit=None,
    )

    assert entries == []


# ---------------------------------------------------------------------------
# limit / max_documents tests
# ---------------------------------------------------------------------------


def test_limit_truncates_result_to_requested_count():
    """Only ``limit`` entries are returned when more are available."""
    fetch = _single_page_fetch(
        _article_html("alpha", "2024-06-01T00:00:00+00:00"),
        _article_html("beta", "2024-05-01T00:00:00+00:00"),
        _article_html("gamma", "2024-04-01T00:00:00+00:00"),
    )

    entries, _ = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=None,
        limit=2,
    )

    assert len(entries) == 2


def test_limit_one_returns_single_entry():
    """limit=1 yields exactly one entry."""
    fetch = _single_page_fetch(
        _article_html("alpha", "2024-06-01T00:00:00+00:00"),
        _article_html("beta", "2024-05-01T00:00:00+00:00"),
    )

    entries, _ = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=None,
        limit=1,
    )

    assert len(entries) == 1


def test_limit_zero_returns_empty_list():
    """limit=0 yields an empty list immediately."""
    fetch = _single_page_fetch(
        _article_html("alpha", "2024-06-01T00:00:00+00:00"),
    )

    entries, _ = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=None,
        limit=0,
    )

    assert entries == []


def test_limit_larger_than_available_returns_all():
    """A limit larger than total results returns all available entries."""
    fetch = _single_page_fetch(
        _article_html("alpha", "2024-06-01T00:00:00+00:00"),
        _article_html("beta", "2024-05-01T00:00:00+00:00"),
    )

    entries, _ = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=None,
        limit=999,
    )

    assert len(entries) == 2


def test_limit_none_returns_all_entries():
    """limit=None does not restrict the result count."""
    fetch = _single_page_fetch(
        _article_html("alpha", "2024-06-01T00:00:00+00:00"),
        _article_html("beta", "2024-05-01T00:00:00+00:00"),
        _article_html("gamma", "2024-04-01T00:00:00+00:00"),
    )

    entries, _ = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=None,
        limit=None,
    )

    assert len(entries) == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_first_page_returns_empty_list():
    """Empty list returned when the first listing page has no articles."""
    fetch = _make_fetch({0: _empty_page()})

    entries, pages_scanned = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=None,
        limit=None,
    )

    assert entries == []
    assert pages_scanned == 1


def test_articles_without_publication_urls_are_ignored():
    """Article cards that do not resolve to a publication URL are skipped."""
    # An anchor that does not match /publications/<slug> pattern
    non_pub_html = (
        "<article>"
        '<a href="/about/team">About</a>'
        '<time datetime="2024-01-01T00:00:00+00:00">2024-01-01</time>'
        "</article>"
    )
    fetch = _make_fetch({0: _listing_page(non_pub_html)})

    entries, _ = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=None,
        limit=None,
    )

    assert entries == []


def test_duplicate_urls_across_pages_are_deduplicated():
    """The same URL appearing on multiple pages produces only one entry."""
    fetch = _make_fetch(
        {
            0: _listing_page(_article_html("dup", "2024-06-01T00:00:00+00:00")),
            1: _listing_page(_article_html("dup", "2024-06-01T00:00:00+00:00")),
        },
    )

    entries, _ = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=None,
        limit=None,
    )

    urls = [e.source_url for e in entries]
    assert urls.count("https://www.lowyinstitute.org/publications/dup") == 1


def test_combined_published_since_and_limit():
    """published_since early-exit and limit both apply; limit caps the result."""
    fetch = _single_page_fetch(
        _article_html("a", "2024-06-01T00:00:00+00:00"),
        _article_html("b", "2024-05-01T00:00:00+00:00"),
        _article_html("c", "2020-01-01T00:00:00+00:00"),  # triggers early exit
    )
    cutoff = datetime(2024, 1, 1, tzinfo=UTC)

    entries, _ = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=cutoff,
        limit=1,
    )

    assert len(entries) == 1


def test_real_lowy_listing_html_inside_card_wrappers_is_parsed():
    """Real listing HTML yields publication URLs from card wrapper entries."""
    html = Path("tmp.html").read_text(encoding="utf-8")
    fetch = _make_fetch({0: html})

    entries, pages_scanned = STRATEGY.discover_documents(
        fetch=fetch,
        published_since=None,
        limit=5,
    )

    assert pages_scanned == 1
    assert len(entries) == 5
    assert entries[0].source_url == (
        "https://www.lowyinstitute.org/publications/"
        "purging-generals-confirms-xi-s-absolute-power"
    )
    assert entries[2].source_url == (
        "https://www.lowyinstitute.org/publications/"
        "inflection-point-biden-trump-future-world-order"
    )
    assert all(entry.published_at is None for entry in entries)
