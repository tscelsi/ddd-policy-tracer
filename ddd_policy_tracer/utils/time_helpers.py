"""Utility functions for handling time-related operations."""

from datetime import UTC, datetime


def utc_now_isoformat() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat()
