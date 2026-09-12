"""Employee directory lookup."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import tool

from .. import db
from .base import fail, ok, safe_tool

EMPLOYEE_ID_PATTERN = re.compile(r"\bEMP\d{4}\b", re.IGNORECASE)


def normalise_employee_id(value: str | None) -> str | None:
    """Return a canonical EMP#### identifier, or None if the text has none."""
    if not value:
        return None
    match = EMPLOYEE_ID_PATTERN.search(value)
    return match.group(0).upper() if match else None


@safe_tool
def _lookup(query: str) -> dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return fail("No employee identifier, email or name was provided.")

    employee_id = normalise_employee_id(query)
    with db.connection() as conn:
        if employee_id:
            row = conn.execute(
                "SELECT * FROM employees WHERE employee_id = ?", (employee_id,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM employees WHERE lower(email) = lower(?)"
                " OR lower(name) = lower(?)",
                (query, query),
            ).fetchone()

    if row is None:
        return fail(
            f"No employee record matches '{query}'. "
            "Employee IDs look like EMP1024."
        )

    record = dict(row)
    record["vpn_enabled"] = bool(record["vpn_enabled"])
    record["mfa_enrolled"] = bool(record["mfa_enrolled"])
    return ok(employee=record)


@tool("get_employee")
def get_employee(query: str) -> dict[str, Any]:
    """Look up an employee profile in the local directory.

    Accepts an employee ID such as EMP1024, a corporate email address, or a
    full name. Use it to confirm that a person exists before acting for them.

    Args:
        query: Employee ID, email address, or full name.
    """
    return _lookup(query)
