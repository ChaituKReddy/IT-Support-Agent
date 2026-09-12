"""Shared result envelope and helpers for every tool.

Tools never raise into the graph. They return a uniform envelope so the agent
can reason about failures and the user interface can render what actually came
back from the local data sources.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def ok(**payload: Any) -> dict[str, Any]:
    """Build a successful tool result."""
    return {"ok": True, "error": None, **payload}


def fail(error: str, **payload: Any) -> dict[str, Any]:
    """Build a failed tool result carrying a user-readable reason."""
    return {"ok": False, "error": error, **payload}


def safe_tool(func: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Convert any unexpected exception into a failed result.

    Without this the agent loop would terminate on a transient database error
    instead of telling the user that a lookup could not be completed.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all boundary
            logger.exception("Tool %s failed", func.__name__)
            return fail(
                f"The {func.__name__} tool could not complete: {exc.__class__.__name__}. "
                "The underlying local data source may be unavailable."
            )

    return wrapper
