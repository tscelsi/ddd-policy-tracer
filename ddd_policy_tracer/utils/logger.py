"""Project logging helpers with lightweight structured context prefixes."""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    _LoggerAdapter = logging.LoggerAdapter[logging.Logger]
else:
    _LoggerAdapter = logging.LoggerAdapter


def configure_logging(level: str) -> None:
    """Configure process-wide logging with compact operator formatting."""
    numeric_level = getattr(logging, level.upper(), logging.WARNING)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )


class CustomLogger(_LoggerAdapter):
    """Prefix log lines with stable context fields for run-level tracing."""

    def process(
        self,
        msg: str,
        kwargs: MutableMapping[str, Any],
    ) -> tuple[str, MutableMapping[str, Any]]:
        """Render context prefix and return the updated message tuple."""
        prefix = _format_context(self.extra or {})
        if not prefix:
            return msg, kwargs
        return f"[{prefix}] {msg}", kwargs

    def bind(self, **extra: str | int | float | bool | None) -> CustomLogger:
        """Return a child logger that extends current context fields."""
        merged = dict(self.extra or {})
        merged.update(extra)
        return CustomLogger(self.logger, merged)


def get_logger(
    name: str,
    **extra: str | int | float | bool | None,
) -> CustomLogger:
    """Create a custom logger bound to one module and optional context."""
    return CustomLogger(logging.getLogger(name), extra)


def _format_context(extra: Mapping[str, Any]) -> str:
    """Render deterministic key=value context pairs for log prefixes."""
    if not extra:
        return ""

    rendered_parts: list[str] = []
    for key in sorted(extra):
        value = extra[key]
        if value is None:
            continue
        rendered_parts.append(f"{key}={value}")
    return " ".join(rendered_parts)


# Backward-compatible alias for existing imports.
CustomLoggingAdapter = CustomLogger
