"""Tool behaviour, including validation and safety rules."""

from __future__ import annotations

from it_support.tools import (
    create_ticket,
    get_employee,
    knowledge_search,
    system_status,
    ticket_lookup,
)


# --- Knowledge search ------------------------------------------------------


def test_knowledge_search_finds_the_vpn_password_article():
    result = knowledge_search.invoke({"query": "How do I reset my VPN password?"})
    assert result["ok"]
    assert result["articles"][0]["article_id"] == "KB-001"


def test_knowledge_search_reports_no_match_instead_of_guessing():
    result = knowledge_search.invoke({"query": "how do I book annual leave in Workday"})
    assert result["ok"]
    assert result["articles"] == []
    assert "Do not invent" in result["note"]


def test_knowledge_search_rejects_an_empty_query():
    result = knowledge_search.invoke({"query": "  "})
    assert not result["ok"]


# --- Employee lookup -------------------------------------------------------


def test_get_employee_by_id_and_by_email():
    assert get_employee.invoke({"query": "EMP1024"})["employee"]["name"] == "Priya Sharma"
    by_email = get_employee.invoke({"query": "daniel.okafor@sourcify.example"})
    assert by_email["employee"]["employee_id"] == "EMP1031"


def test_get_employee_reports_an_unknown_person():
    result = get_employee.invoke({"query": "EMP9999"})
    assert not result["ok"]
    assert "EMP1024" in result["error"]


# --- Ticket lookup ---------------------------------------------------------


def test_ticket_lookup_by_employee_returns_their_tickets_only():
    result = ticket_lookup.invoke({"employee_id": "EMP1024"})
    assert result["ok"]
    assert result["count"] == 2
    assert {t["employee_id"] for t in result["tickets"]} == {"EMP1024"}


def test_ticket_lookup_by_ticket_id_normalises_the_reference():
    result = ticket_lookup.invoke({"ticket_id": "10001"})
    assert result["tickets"][0]["ticket_id"] == "INC-10001"


def test_ticket_lookup_by_keyword_and_status():
    result = ticket_lookup.invoke({"keyword": "laptop", "status": "open"})
    assert all(t["status"] == "Open" for t in result["tickets"])


def test_ticket_lookup_needs_at_least_one_filter():
    assert not ticket_lookup.invoke({})["ok"]


def test_ticket_lookup_rejects_a_malformed_employee_id():
    result = ticket_lookup.invoke({"employee_id": "not-an-id"})
    assert not result["ok"]


def test_ticket_lookup_reports_no_results_without_inventing_any():
    result = ticket_lookup.invoke({"employee_id": "EMP1088", "status": "Closed"})
    assert result["ok"]
    assert result["tickets"] == []
    assert result["note"]


# --- Ticket creation -------------------------------------------------------


VALID_TICKET = {
    "employee_id": "EMP1031",
    "category": "Network",
    "subject": "Wi-Fi drops in the Dublin office",
    "description": "The corporate Wi-Fi disconnects roughly twice an hour.",
    "priority": "High",
}


def test_create_ticket_writes_a_new_ticket():
    result = create_ticket.invoke(dict(VALID_TICKET))
    assert result["ok"] and result["created"]
    ticket = result["ticket"]
    assert ticket["ticket_id"].startswith("INC-")
    assert ticket["status"] == "Open"

    stored = ticket_lookup.invoke({"ticket_id": ticket["ticket_id"]})
    assert stored["tickets"][0]["subject"] == VALID_TICKET["subject"]


def test_create_ticket_refuses_to_duplicate_an_open_ticket():
    first = create_ticket.invoke(dict(VALID_TICKET))
    second = create_ticket.invoke(
        {**VALID_TICKET, "subject": "Dublin office wi-fi keeps dropping"}
    )
    assert second["ok"]
    assert second["created"] is False
    assert second["duplicate_of"]["ticket_id"] == first["ticket"]["ticket_id"]


def test_create_ticket_reports_every_missing_required_field():
    result = create_ticket.invoke(
        {"employee_id": "", "category": "VPN", "subject": "", "description": ""}
    )
    assert not result["ok"]
    assert result["missing_fields"] == ["employee_id", "subject", "description"]


def test_create_ticket_rejects_an_unknown_employee():
    result = create_ticket.invoke({**VALID_TICKET, "employee_id": "EMP9999"})
    assert not result["ok"]
    assert "no ticket was created" in result["error"].lower()


def test_create_ticket_rejects_an_invalid_category():
    result = create_ticket.invoke({**VALID_TICKET, "category": "Teleportation"})
    assert not result["ok"]
    assert result["missing_fields"] == ["category"]


def test_create_ticket_rejects_an_invalid_priority():
    result = create_ticket.invoke({**VALID_TICKET, "priority": "Yesterday"})
    assert not result["ok"]
    assert result["missing_fields"] == ["priority"]


def test_create_ticket_accepts_lowercase_category_and_priority():
    result = create_ticket.invoke({**VALID_TICKET, "category": "network", "priority": "high"})
    assert result["created"]
    assert result["ticket"]["category"] == "Network"
    assert result["ticket"]["priority"] == "High"


# --- System status ---------------------------------------------------------


def test_system_status_filters_by_service():
    result = system_status.invoke({"service": "vpn"})
    assert result["services"][0]["service"] == "VPN Gateway"


def test_system_status_lists_degraded_services():
    result = system_status.invoke({})
    assert "Print Services" in result["degraded"]


def test_system_status_handles_an_unknown_service():
    result = system_status.invoke({"service": "mainframe"})
    assert result["ok"] and result["services"] == []


# --- Failure handling ------------------------------------------------------


def test_tool_failure_is_reported_rather_than_raised(monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("database is gone")

    monkeypatch.setattr("it_support.tools.employee.db.connection", explode)
    result = get_employee.invoke({"query": "EMP1024"})
    assert not result["ok"]
    assert "could not complete" in result["error"]
