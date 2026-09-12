"""Ticket lookup and ticket creation tools (Tools 2 and 3)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from langchain_core.tools import tool

from .. import db
from ..config import get_settings
from ..search import tokenize
from .base import fail, ok, safe_tool
from .employee import normalise_employee_id

OPEN_STATUSES = ("Open", "In Progress")

VALID_PRIORITIES = ("Low", "Medium", "High", "Critical")
VALID_CATEGORIES = (
    "VPN",
    "Network",
    "Hardware",
    "Software",
    "Email",
    "Access",
    "Account",
    "Password",
    "Printer",
    "Security",
    "Onboarding",
    "Other",
)

# Fraction of overlapping subject words above which two open tickets from the
# same employee are treated as the same problem.
DUPLICATE_SIMILARITY_THRESHOLD = 0.5

TICKET_ID_PREFIX = "INC-"


def _row_to_ticket(row) -> dict[str, Any]:
    return dict(row)


def _canonical(value: str, allowed: tuple[str, ...]) -> str | None:
    """Match user text against an allowed vocabulary, ignoring case."""
    lowered = (value or "").strip().lower()
    for option in allowed:
        if option.lower() == lowered:
            return option
    return None


def subject_similarity(left: str, right: str) -> float:
    """Jaccard overlap of the meaningful words in two ticket subjects."""
    left_tokens, right_tokens = set(tokenize(left)), set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


# ---------------------------------------------------------------------------
# Tool 2: ticket lookup
# ---------------------------------------------------------------------------


@safe_tool
def _lookup(
    employee_id: str | None = None,
    ticket_id: str | None = None,
    status: str | None = None,
    keyword: str | None = None,
) -> dict[str, Any]:
    if not any([employee_id, ticket_id, status, keyword]):
        return fail(
            "A ticket lookup needs at least an employee ID, a ticket ID, "
            "a status, or a keyword."
        )

    clauses: list[str] = []
    params: list[Any] = []

    if ticket_id:
        normalised = ticket_id.strip().upper()
        if not normalised.startswith(TICKET_ID_PREFIX):
            normalised = f"{TICKET_ID_PREFIX}{normalised.lstrip('#')}"
        clauses.append("ticket_id = ?")
        params.append(normalised)

    if employee_id:
        canonical_employee = normalise_employee_id(employee_id)
        if canonical_employee is None:
            return fail(
                f"'{employee_id}' is not a valid employee ID. "
                "Employee IDs look like EMP1024."
            )
        clauses.append("employee_id = ?")
        params.append(canonical_employee)

    if status:
        clauses.append("lower(status) = lower(?)")
        params.append(status.strip())

    if keyword:
        clauses.append("(subject LIKE ? OR description LIKE ? OR category LIKE ?)")
        pattern = f"%{keyword.strip()}%"
        params.extend([pattern, pattern, pattern])

    sql = (
        "SELECT * FROM tickets WHERE "
        + " AND ".join(clauses)
        + " ORDER BY created_at DESC LIMIT 25"
    )

    with db.connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    tickets = [_row_to_ticket(row) for row in rows]
    return ok(
        tickets=tickets,
        count=len(tickets),
        note=(
            "No matching ticket exists in the local system."
            if not tickets
            else None
        ),
    )


@tool("ticket_lookup")
def ticket_lookup(
    employee_id: str | None = None,
    ticket_id: str | None = None,
    status: str | None = None,
    keyword: str | None = None,
) -> dict[str, Any]:
    """Search existing support tickets in the local ticket system.

    Use this to answer questions about the status or history of an issue, and
    always before creating a ticket so that duplicates can be spotted.

    Args:
        employee_id: Owner of the tickets, for example EMP1024.
        ticket_id: A specific ticket reference, for example INC-10001.
        status: Filter by status: Open, In Progress, Resolved, or Closed.
        keyword: Free text matched against the subject, description and category.
    """
    return _lookup(
        employee_id=employee_id, ticket_id=ticket_id, status=status, keyword=keyword
    )


# ---------------------------------------------------------------------------
# Tool 3: ticket creation
# ---------------------------------------------------------------------------


def _next_ticket_id(conn) -> str:
    row = conn.execute("SELECT MAX(CAST(SUBSTR(ticket_id, 5) AS INTEGER)) FROM tickets")
    highest = row.fetchone()[0] or 10000
    return f"{TICKET_ID_PREFIX}{highest + 1}"


def find_duplicate(conn, employee_id: str, subject: str, category: str) -> dict | None:
    """Return an existing open ticket that describes the same problem, if any."""
    settings = get_settings()
    window_start = (
        datetime.now() - timedelta(days=settings.duplicate_ticket_window_days)
    ).isoformat()

    placeholders = ", ".join("?" for _ in OPEN_STATUSES)
    rows = conn.execute(
        f"""SELECT * FROM tickets
            WHERE employee_id = ?
              AND status IN ({placeholders})
              AND created_at >= ?""",
        (employee_id, *OPEN_STATUSES, window_start),
    ).fetchall()

    for row in rows:
        same_category = row["category"].lower() == category.lower()
        similar_subject = (
            subject_similarity(row["subject"], subject) >= DUPLICATE_SIMILARITY_THRESHOLD
        )
        if same_category or similar_subject:
            return _row_to_ticket(row)
    return None


@safe_tool
def _create(
    employee_id: str,
    category: str,
    subject: str,
    description: str,
    priority: str = "Medium",
) -> dict[str, Any]:
    missing = [
        name
        for name, value in (
            ("employee_id", employee_id),
            ("category", category),
            ("subject", subject),
            ("description", description),
        )
        if not (value or "").strip()
    ]
    if missing:
        return fail(
            "Cannot create a ticket yet, required details are missing: "
            + ", ".join(missing),
            missing_fields=missing,
        )

    canonical_employee = normalise_employee_id(employee_id)
    if canonical_employee is None:
        return fail(
            f"'{employee_id}' is not a valid employee ID. "
            "Employee IDs look like EMP1024.",
            missing_fields=["employee_id"],
        )

    canonical_category = _canonical(category, VALID_CATEGORIES)
    if canonical_category is None:
        return fail(
            f"'{category}' is not a valid category. Choose one of: "
            + ", ".join(VALID_CATEGORIES),
            missing_fields=["category"],
        )

    canonical_priority = _canonical(priority, VALID_PRIORITIES)
    if canonical_priority is None:
        return fail(
            f"'{priority}' is not a valid priority. Choose one of: "
            + ", ".join(VALID_PRIORITIES),
            missing_fields=["priority"],
        )

    with db.connection() as conn:
        employee = conn.execute(
            "SELECT * FROM employees WHERE employee_id = ?", (canonical_employee,)
        ).fetchone()
        if employee is None:
            return fail(
                f"No employee record exists for {canonical_employee}, so no ticket "
                "was created. Please confirm the employee ID."
            )

        duplicate = find_duplicate(
            conn, canonical_employee, subject, canonical_category
        )
        if duplicate is not None:
            return ok(
                created=False,
                duplicate_of=duplicate,
                note=(
                    f"An open {duplicate['category']} ticket "
                    f"({duplicate['ticket_id']}) already exists for this employee. "
                    "No new ticket was created. Ask the employee whether they want "
                    "to add to that ticket or raise a separate one anyway."
                ),
            )

        now = datetime.now().isoformat(timespec="seconds")
        ticket_id = _next_ticket_id(conn)
        conn.execute(
            """INSERT INTO tickets
               (ticket_id, employee_id, category, priority, status, subject,
                description, assigned_to, created_at, updated_at, resolution)
               VALUES (?, ?, ?, ?, 'Open', ?, ?, 'Service Desk', ?, ?, NULL)""",
            (
                ticket_id,
                canonical_employee,
                canonical_category,
                canonical_priority,
                subject.strip(),
                description.strip(),
                now,
                now,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()

    return ok(created=True, ticket=_row_to_ticket(row))


@tool("create_ticket")
def create_ticket(
    employee_id: str,
    category: str,
    subject: str,
    description: str,
    priority: str = "Medium",
) -> dict[str, Any]:
    """Create a new support ticket in the local ticket system.

    Only call this once the employee has confirmed they want a ticket raised and
    every argument is known. If an equivalent open ticket already exists, no new
    ticket is created and the existing one is returned instead.

    Args:
        employee_id: Owner of the ticket, for example EMP1024.
        category: One of VPN, Network, Hardware, Software, Email, Access,
            Account, Password, Printer, Security, Onboarding, Other.
        subject: A short one-line summary of the problem.
        description: What the employee reported, in detail.
        priority: Low, Medium, High or Critical. Defaults to Medium.
    """
    return _create(
        employee_id=employee_id,
        category=category,
        subject=subject,
        description=description,
        priority=priority,
    )
