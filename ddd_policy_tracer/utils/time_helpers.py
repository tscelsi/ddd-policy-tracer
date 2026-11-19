"""Utility functions for handling time-related operations."""

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

_CANBERRA = ZoneInfo("Australia/Canberra")


def utc_now_isoformat() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def utc_isoformat(value: datetime) -> str:
    """Serialize a timestamp as UTC, treating naive database timestamps as UTC."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def utc_published_at_bounds(
    *,
    date_from: date | None,
    date_to: date | None,
) -> tuple[str | None, str | None]:
    """Convert inclusive Canberra calendar dates to UTC ISO bounds."""
    start = canberra_midnight_utc(date_from) if date_from is not None else None
    end = canberra_midnight_utc(date_to + timedelta(days=1)) if date_to is not None else None
    return iso_utc(start), iso_utc(end)


def canberra_midnight_utc(day: date) -> datetime:
    """Map a local calendar midnight to its UTC instant."""
    return datetime.combine(day, time.min, tzinfo=_CANBERRA).astimezone(UTC)


def iso_utc(value: datetime | None) -> str | None:
    """Serialize a UTC instant in the format stored for Hansard speech dates."""
    return value.isoformat() if value is not None else None
