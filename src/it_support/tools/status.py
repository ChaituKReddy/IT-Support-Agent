"""System status tool: current health of the internal IT services."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from .. import db
from .base import ok, safe_tool


@safe_tool
def _status(service: str | None = None) -> dict[str, Any]:
    with db.connection() as conn:
        if service and service.strip():
            rows = conn.execute(
                "SELECT * FROM system_status WHERE lower(service) LIKE lower(?)",
                (f"%{service.strip()}%",),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM system_status ORDER BY service").fetchall()

    services = [dict(row) for row in rows]
    if not services:
        return ok(
            services=[],
            note=f"No monitored service matches '{service}'.",
        )

    return ok(
        services=services,
        degraded=[s["service"] for s in services if s["status"] != "Operational"],
    )


@tool("system_status")
def system_status(service: str | None = None) -> dict[str, Any]:
    """Check the current status of internal IT services.

    Use this when an employee reports an outage or a slow service, so that a
    known incident is not duplicated as a new ticket.

    Args:
        service: Optional service name filter, for example VPN, Email, Print.
    """
    return _status(service)
