"""Conversation state carried through the LangGraph workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Route = Literal["tools", "confirm", "clarify", "respond"]


class AgentState(TypedDict, total=False):
    """Everything the assistant remembers for one conversation thread.

    ``messages`` is the running transcript. The remaining fields are the slots
    the assistant fills in as the conversation progresses, so that a follow-up
    such as "yes, go ahead" can be understood without repeating earlier details.
    """

    messages: Annotated[list[AnyMessage], add_messages]

    # Identity slots, remembered once the employee has been identified.
    employee_id: str | None
    employee_profile: dict[str, Any] | None

    # A ticket the assistant has proposed but not yet created.
    pending_ticket: dict[str, Any] | None
    #: True only for the single turn in which the user confirmed the ticket.
    confirmed: bool
    #: Fields the assistant still needs before it can act.
    missing_fields: list[str]

    # Observability: every tool call and its result, for the UI trace panel.
    tool_events: Annotated[list[dict[str, Any]], operator.add]

    # Loop control.
    steps: int
    route: Route


def initial_state() -> AgentState:
    """A blank state for a brand new conversation thread."""
    return AgentState(
        messages=[],
        employee_id=None,
        employee_profile=None,
        pending_ticket=None,
        confirmed=False,
        missing_fields=[],
        tool_events=[],
        steps=0,
        route="respond",
    )
