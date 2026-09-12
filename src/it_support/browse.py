"""Read-only queries used by the user interface reference panels.

These are deliberately separate from the agent's tools: the panels let a human
browse the same local data the assistant works from, without going through the
model.
"""

from __future__ import annotations

from typing import Any

from . import db
from .search import build_index

TICKET_STATUSES = ("Open", "In Progress", "Resolved", "Closed")


def list_articles(query: str = "") -> list[dict[str, Any]]:
    """All knowledge base articles, ranked by relevance when a query is given."""
    with db.connection() as conn:
        rows = [dict(row) for row in conn.execute("SELECT * FROM kb_articles")]

    if not query.strip():
        return sorted(rows, key=lambda row: (row["category"], row["article_id"]))

    hits = build_index(rows).search(query, top_k=len(rows))
    by_id = {row["article_id"]: row for row in rows}
    return [{**by_id[hit.document.article_id], "relevance": hit.score} for hit in hits]


def list_tickets(
    status: str = "", category: str = "", employee_id: str = ""
) -> list[dict[str, Any]]:
    """Tickets from the local system, newest first, with optional filters."""
    clauses: list[str] = []
    params: list[Any] = []
    for column, value in (
        ("status", status),
        ("category", category),
        ("employee_id", employee_id),
    ):
        if value:
            clauses.append(f"{column} = ?")
            params.append(value)

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with db.connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM tickets{where} ORDER BY created_at DESC", params
        ).fetchall()
    return [dict(row) for row in rows]


def ticket_categories() -> list[str]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT category FROM tickets ORDER BY category"
        ).fetchall()
    return [row["category"] for row in rows]


def list_employees() -> list[dict[str, Any]]:
    with db.connection() as conn:
        rows = conn.execute("SELECT * FROM employees ORDER BY employee_id").fetchall()
    return [dict(row) for row in rows]


def ticket_counts() -> dict[str, int]:
    """How many tickets sit in each status, for the summary strip."""
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS total FROM tickets GROUP BY status"
        ).fetchall()
    counts = {row["status"]: row["total"] for row in rows}
    return {status: counts.get(status, 0) for status in TICKET_STATUSES}


def service_status() -> list[dict[str, Any]]:
    with db.connection() as conn:
        rows = conn.execute("SELECT * FROM system_status ORDER BY service").fetchall()
    return [dict(row) for row in rows]
