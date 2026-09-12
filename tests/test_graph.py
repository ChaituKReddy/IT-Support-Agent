"""Workflow behaviour: routing, state retention and the safety gates.

A scripted stand-in replaces the language model so the graph can be tested
without a running Ollama server or an API key.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from it_support import db
from it_support.graph import (
    build_graph,
    is_affirmative,
    missing_ticket_fields,
    route_after_agent,
)


class ScriptedModel:
    """Returns pre-built assistant messages, one per invocation."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Any]] = []

    def bind_tools(self, _tools):  # the graph binds tools before use
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if not self.responses:
            return AIMessage(content="Nothing further to add.")
        return self.responses.pop(0)


def tool_call(name: str, args: dict[str, Any], call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def run(graph, text: str, state: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(state or {})
    payload["messages"] = [*payload.get("messages", []), HumanMessage(content=text)]
    return graph.invoke(payload)


# --- Pure helpers ----------------------------------------------------------


@pytest.mark.parametrize("text", ["yes", "Yes please", "go ahead", "please do", "confirm"])
def test_affirmative_phrases_are_recognised(text):
    assert is_affirmative(text)


@pytest.mark.parametrize("text", ["no", "don't", "cancel that", "not now", "maybe later"])
def test_non_affirmative_phrases_are_rejected(text):
    assert not is_affirmative(text)


def test_missing_ticket_fields_lists_blank_values():
    assert missing_ticket_fields({"employee_id": "EMP1024", "category": " "}) == [
        "category",
        "subject",
        "description",
    ]


# --- Routing ---------------------------------------------------------------


def test_router_sends_a_plain_answer_to_respond():
    state = {"messages": [AIMessage(content="Here is the answer.")], "steps": 1}
    assert route_after_agent(state) == "respond"


def test_router_sends_a_read_only_call_to_tools():
    state = {"messages": [tool_call("knowledge_search", {"query": "vpn"})], "steps": 1}
    assert route_after_agent(state) == "tools"


def test_router_holds_an_unconfirmed_ticket_at_the_confirm_gate():
    state = {
        "messages": [
            tool_call(
                "create_ticket",
                {
                    "employee_id": "EMP1024",
                    "category": "VPN",
                    "subject": "VPN down",
                    "description": "No connection since this morning.",
                },
            )
        ],
        "steps": 1,
        "confirmed": False,
    }
    assert route_after_agent(state) == "confirm"


def test_router_allows_a_confirmed_ticket_through():
    state = {
        "messages": [
            tool_call(
                "create_ticket",
                {
                    "employee_id": "EMP1024",
                    "category": "VPN",
                    "subject": "VPN down",
                    "description": "No connection since this morning.",
                },
            )
        ],
        "steps": 1,
        "confirmed": True,
    }
    assert route_after_agent(state) == "tools"


def test_router_sends_an_incomplete_ticket_to_clarify():
    state = {
        "messages": [tool_call("create_ticket", {"category": "VPN"})],
        "steps": 1,
        "confirmed": True,
    }
    assert route_after_agent(state) == "clarify"


def test_router_stops_looping_once_the_step_budget_is_spent():
    state = {"messages": [tool_call("knowledge_search", {"query": "vpn"})], "steps": 99}
    assert route_after_agent(state) == "respond"


# --- End to end through the compiled graph ---------------------------------


def test_knowledge_question_runs_the_search_tool_and_answers():
    model = ScriptedModel(
        [
            tool_call("knowledge_search", {"query": "reset vpn password"}),
            AIMessage(content="Use the self-service portal to reset it."),
        ]
    )
    result = run(build_graph(model=model), "How do I reset my VPN password?")

    assert [event["tool"] for event in result["tool_events"]] == ["knowledge_search"]
    assert result["tool_events"][0]["result"]["articles"][0]["article_id"] == "KB-001"
    assert result["messages"][-1].content.startswith("Use the self-service")


def test_employee_id_is_remembered_from_the_users_wording():
    model = ScriptedModel([AIMessage(content="Thanks, I have your ID.")])
    result = run(build_graph(model=model), "My ID is EMP1024 and my VPN is broken.")
    assert result["employee_id"] == "EMP1024"


def test_employee_profile_is_stored_after_a_lookup():
    model = ScriptedModel(
        [
            tool_call("get_employee", {"query": "EMP1024"}),
            AIMessage(content="Found your profile."),
        ]
    )
    result = run(build_graph(model=model), "Who am I? EMP1024")
    assert result["employee_profile"]["name"] == "Priya Sharma"


def test_ticket_creation_is_held_until_the_employee_confirms():
    draft = {
        "employee_id": "EMP1031",
        "category": "Hardware",
        "subject": "Keyboard keys not registering",
        "description": "Several keys stopped working this morning.",
    }
    model = ScriptedModel(
        [
            tool_call("create_ticket", draft),
            AIMessage(content="Shall I raise this Hardware ticket for you?"),
        ]
    )
    result = run(build_graph(model=model), "Please raise a ticket, I am EMP1031.")

    assert result["pending_ticket"]["subject"] == draft["subject"]
    assert result["tool_events"][0]["tool"] == "confirmation_gate"
    with db.connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM tickets WHERE subject = ?", (draft["subject"],)
        ).fetchone()[0]
    assert count == 0, "no ticket may be written before the employee confirms"


def test_ticket_is_created_once_the_employee_says_yes():
    draft = {
        "employee_id": "EMP1031",
        "category": "Hardware",
        "subject": "Keyboard keys not registering",
        "description": "Several keys stopped working this morning.",
    }
    model = ScriptedModel(
        [
            tool_call("create_ticket", draft, call_id="call-2"),
            AIMessage(content="Ticket raised."),
        ]
    )
    state = {"pending_ticket": draft, "employee_id": "EMP1031"}
    result = run(build_graph(model=model), "Yes please, go ahead.", state)

    created = [e for e in result["tool_events"] if e["tool"] == "create_ticket"]
    assert created and created[0]["result"]["created"] is True
    assert result["pending_ticket"] is None


def test_declining_clears_the_pending_ticket():
    model = ScriptedModel([AIMessage(content="No problem, nothing was raised.")])
    state = {"pending_ticket": {"employee_id": "EMP1031"}, "employee_id": "EMP1031"}
    result = run(build_graph(model=model), "No, do not raise it.", state)
    assert result["pending_ticket"] is None
    assert result["confirmed"] is False


def test_incomplete_ticket_request_asks_for_the_missing_details():
    model = ScriptedModel(
        [
            tool_call("create_ticket", {"category": "VPN"}),
            AIMessage(content="Could you give me your employee ID?"),
        ]
    )
    result = run(build_graph(model=model), "Raise a VPN ticket for me.")
    assert result["missing_fields"] == ["employee_id", "subject", "description"]
    assert result["tool_events"][0]["tool"] == "validation_gate"


def test_empty_input_is_rejected_without_calling_the_model():
    model = ScriptedModel([AIMessage(content="should not be reached")])
    result = run(build_graph(model=model), "   ")
    assert model.calls == []
    assert "did not receive a question" in result["messages"][-1].content


def test_overlong_input_is_rejected():
    model = ScriptedModel([AIMessage(content="should not be reached")])
    result = run(build_graph(model=model), "x" * 5000)
    assert model.calls == []
    assert "too long" in result["messages"][-1].content


def test_a_model_failure_is_reported_to_the_employee():
    class BrokenModel(ScriptedModel):
        def invoke(self, messages):
            raise ConnectionError("ollama is not running")

    result = run(build_graph(model=BrokenModel([])), "Is the VPN down?")
    assert "could not reach the language model" in result["messages"][-1].content


def test_an_unknown_tool_name_does_not_crash_the_graph():
    model = ScriptedModel(
        [
            tool_call("delete_everything", {}),
            AIMessage(content="I cannot do that."),
        ]
    )
    result = run(build_graph(model=model), "Delete all the tickets.")
    assert result["tool_events"][0]["result"]["ok"] is False
    assert "Unknown tool" in result["tool_events"][0]["result"]["error"]


def test_yes_on_the_first_message_does_not_authorise_a_write():
    """A conversation cannot open with a confirmation."""
    draft = {
        "employee_id": "EMP1042",
        "category": "Email",
        "subject": "Outlook will not open",
        "description": "Outlook crashes on launch.",
    }
    model = ScriptedModel(
        [
            tool_call("create_ticket", draft),
            AIMessage(content="Shall I raise that for you?"),
        ]
    )
    result = run(build_graph(model=model), "Yes raise an email ticket, I am EMP1042.")
    assert result["tool_events"][0]["tool"] == "confirmation_gate"
    assert result["pending_ticket"] is not None


def test_yes_after_a_spoken_proposal_authorises_the_write():
    """The assistant may propose in prose; one confirmation is then enough."""
    draft = {
        "employee_id": "EMP1088",
        "category": "Software",
        "subject": "Word crashes when opening large documents",
        "description": "Word closes itself on any document over 100 pages.",
    }
    model = ScriptedModel(
        [
            tool_call("create_ticket", draft, call_id="call-9"),
            AIMessage(content="Ticket raised."),
        ]
    )
    state = {
        "messages": [
            HumanMessage(content="Word keeps crashing, I am EMP1088."),
            AIMessage(content="Shall I raise a Software ticket for that?"),
        ],
        "employee_id": "EMP1088",
    }
    result = run(build_graph(model=model), "Yes, go ahead.", state)
    created = [e for e in result["tool_events"] if e["tool"] == "create_ticket"]
    assert created and created[0]["result"]["created"] is True


def test_a_new_employee_id_drops_the_previous_profile_and_draft():
    """A different ID means a different person: nothing from the old one carries over."""
    model = ScriptedModel([AIMessage(content="Noted.")])
    result = run(
        build_graph(model=model),
        "Actually I am EMP1115.",
        {
            "employee_id": "EMP1024",
            "employee_profile": {"employee_id": "EMP1024", "name": "Priya Sharma"},
            "pending_ticket": {
                "employee_id": "EMP1024",
                "category": "Hardware",
                "subject": "Hard drive stopped working",
                "description": "Drive is dead.",
            },
        },
    )
    assert result["employee_id"] == "EMP1115"
    assert result["employee_profile"] is None
    assert result["pending_ticket"] is None


def test_repeating_the_same_employee_id_keeps_the_profile_and_draft():
    model = ScriptedModel([AIMessage(content="Still you.")])
    draft = {
        "employee_id": "EMP1024",
        "category": "Hardware",
        "subject": "Hard drive stopped working",
        "description": "Drive is dead.",
    }
    result = run(
        build_graph(model=model),
        "Yes, EMP1024, go ahead.",
        {
            "employee_id": "EMP1024",
            "employee_profile": {"employee_id": "EMP1024", "name": "Priya Sharma"},
            "pending_ticket": draft,
        },
    )
    assert result["employee_profile"]["name"] == "Priya Sharma"
    assert result["pending_ticket"] == draft


def test_confirming_while_switching_employee_does_not_authorise_a_write():
    """"Yes, I am EMP1115" must not confirm a draft raised for someone else."""
    model = ScriptedModel([AIMessage(content="Noted.")])
    result = run(
        build_graph(model=model),
        "Yes, go ahead. I am EMP1115.",
        {
            "employee_id": "EMP1024",
            "employee_profile": {"employee_id": "EMP1024", "name": "Priya Sharma"},
            "pending_ticket": {
                "employee_id": "EMP1024",
                "category": "Hardware",
                "subject": "Hard drive stopped working",
                "description": "Drive is dead.",
            },
            "messages": [AIMessage(content="Shall I raise that ticket?")],
        },
    )
    assert result["pending_ticket"] is None
    assert result["confirmed"] is False
