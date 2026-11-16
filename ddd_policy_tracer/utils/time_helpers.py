"""Utility functions for handling time-related operations."""

from datetime import UTC, datetime


def utc_now_isoformat() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()


def utc_isoformat(value: datetime) -> str:
    """Serialize a timestamp as UTC, treating naive database timestamps as UTC."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()
