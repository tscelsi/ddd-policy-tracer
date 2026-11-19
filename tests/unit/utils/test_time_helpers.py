from datetime import date

from ddd_policy_tracer.utils.time_helpers import utc_published_at_bounds


def test_utc_published_at_bounds_cover_inclusive_canberra_dates() -> None:
    """Convert inclusive Australian calendar dates into UTC timestamp bounds."""
    start, end = utc_published_at_bounds(
        date_from=date(2025, 1, 1),
        date_to=date(2025, 1, 1),
    )

    assert start == "2024-12-31T13:00:00+00:00"
    assert end == "2025-01-01T13:00:00+00:00"


def test_utc_published_at_bounds_support_open_ended_ranges() -> None:
    """Leave either UTC bound unset for one-sided date requests."""
    assert utc_published_at_bounds(date_from=None, date_to=date(2024, 6, 30)) == (
        None,
        "2024-06-30T14:00:00+00:00",
    )
    assert utc_published_at_bounds(date_from=date(2024, 7, 1), date_to=None) == (
        "2024-06-30T14:00:00+00:00",
        None,
    )
