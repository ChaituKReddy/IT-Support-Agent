"""LangGraph workflow for the IT support assistant.

    START -> validate -> agent -> route
                          ^        |-- tools   -> agent
                          |        |-- confirm -> respond
                          +--------+-- clarify -> respond
                                   +-- respond -> END

``validate`` normalises the incoming turn and updates the remembered slots,
``agent`` decides whether a tool is needed, the conditional edge routes that
decision, and ``respond`` produces the final user-facing answer.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from typing import Any, Iterable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from .config import Settings, get_settings
from .llm import build_agent_model
from .prompts import (
    CONFIRMATION_REQUEST,
    CONVERSATION_STATE_TEMPLATE,
    MISSING_INFORMATION,
    STEP_LIMIT_REACHED,
    SYSTEM_PROMPT,
    format_state_block,
)
from .state import AgentState, initial_state
from .tools import ALL_TOOLS, TOOLS_BY_NAME, WRITE_TOOLS
from .tools.employee import normalise_employee_id
from .tools.tickets import VALID_CATEGORIES

logger = logging.getLogger(__name__)

REQUIRED_TICKET_FIELDS = ("employee_id", "category", "subject", "description")

_AFFIRMATIVE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok|okay|please\s+do|go\s+ahead|do\s+it|"
    r"raise\s+it|create\s+it|confirm(ed)?|proceed)\b",
    re.IGNORECASE,
)
_NEGATIVE = re.compile(
    r"\b(no|nope|don'?t|do\s+not|cancel|stop|never\s*mind|not\s+now)\b",
    re.IGNORECASE,
)

MAX_INPUT_CHARACTERS = 2000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _last_human_text(messages: Iterable[Any]) -> str:
    for message in reversed(list(messages)):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def is_affirmative(text: str) -> bool:
    """True when the user has clearly agreed to the pending action."""
    if _NEGATIVE.search(text):
        return False
    return bool(_AFFIRMATIVE.search(text))


def _write_tool_calls(message: Any) -> list[dict[str, Any]]:
    calls = getattr(message, "tool_calls", None) or []
    return [call for call in calls if call["name"] in WRITE_TOOLS]


def missing_ticket_fields(args: dict[str, Any]) -> list[str]:
    """Required create_ticket arguments that are absent or blank."""
    return [
        field
        for field in REQUIRED_TICKET_FIELDS
        if not str(args.get(field) or "").strip()
    ]


def _merge_draft(
    pending: dict[str, Any] | None,
    args: dict[str, Any],
    employee_id: str | None,
) -> dict[str, Any]:
    """Combine a previously proposed ticket with newly supplied arguments."""
    draft: dict[str, Any] = dict(pending or {})
    draft.update({k: v for k, v in args.items() if str(v or "").strip()})
    if employee_id and not draft.get("employee_id"):
        draft["employee_id"] = employee_id
    return draft


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def validate_node(state: AgentState) -> dict[str, Any]:
    """Guard the incoming turn and refresh the remembered slots.

    This node is deterministic: it never calls the model. It extracts an
    employee ID from the user's wording, recognises an explicit confirmation of
    a pending ticket, and rejects input that is empty or unreasonably long.
    """
    text = _last_human_text(state.get("messages", []))
    updates: dict[str, Any] = {"steps": 0, "missing_fields": []}

    if not text.strip():
        updates["route"] = "respond"
        updates["messages"] = [
            AIMessage(
                content="I did not receive a question. Please describe your IT issue."
            )
        ]
        return updates

    if len(text) > MAX_INPUT_CHARACTERS:
        updates["route"] = "respond"
        updates["messages"] = [
            AIMessage(
                content=(
                    "That message is too long for me to process. Please summarise "
                    f"the problem in under {MAX_INPUT_CHARACTERS} characters."
                )
            )
        ]
        return updates

    employee_id = normalise_employee_id(text)
    if employee_id and employee_id != state.get("employee_id"):
        # The conversation has switched to a different person. The remembered
        # profile and any half-built draft belong to the previous employee, so
        # they are dropped rather than attributed to the new one.
        updates["employee_id"] = employee_id
        updates["employee_profile"] = None
        if state.get("pending_ticket"):
            updates["pending_ticket"] = None
            updates["confirmed"] = False

    # "pending_ticket" in updates means the draft was just dropped above.
    pending = None if "pending_ticket" in updates else state.get("pending_ticket")
    if pending and _NEGATIVE.search(text):
        updates["pending_ticket"] = None
        updates["confirmed"] = False
        return updates

    # A confirmation only counts when there is something to confirm: either a
    # ticket the gate has already held back, or a proposal the assistant made
    # in its previous turn. This keeps "yes" from authorising a write on the
    # very first message of a conversation.
    replying_to_assistant = any(
        isinstance(message, AIMessage) for message in state.get("messages", [])
    )
    identity_changed = "employee_profile" in updates
    updates["confirmed"] = (
        is_affirmative(text)
        and bool(pending or replying_to_assistant)
        # A "yes" that also names a new employee re-opens the question of who
        # the ticket is for; it cannot stand as consent for the old proposal.
        and not identity_changed
    )

    return updates


def make_agent_node(model: BaseChatModel):
    """Build the node that asks the model what to do next."""

    def agent_node(state: AgentState) -> dict[str, Any]:
        if state.get("route") == "respond" and state.get("steps", 0) == 0:
            # validate_node already answered (empty or oversized input).
            last = state["messages"][-1] if state.get("messages") else None
            if isinstance(last, AIMessage):
                return {}

        system = SYSTEM_PROMPT + "\n" + CONVERSATION_STATE_TEMPLATE.format(
            state_block=format_state_block(
                state.get("employee_id"),
                state.get("employee_profile"),
                state.get("pending_ticket"),
            )
        )
        if state.get("confirmed") and state.get("pending_ticket"):
            system += (
                "\nThe employee has just confirmed the pending ticket. Call "
                "create_ticket now with exactly the draft details above."
            )

        messages = [SystemMessage(content=system), *state["messages"]]
        try:
            response = model.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - the model is an external service
            logger.exception("Chat model call failed")
            return {
                "route": "respond",
                "messages": [
                    AIMessage(
                        content=(
                            "I could not reach the language model "
                            f"({exc.__class__.__name__}). Please check that the "
                            "configured provider is running and try again."
                        )
                    )
                ],
            }

        return {"messages": [response], "steps": state.get("steps", 0) + 1}

    return agent_node


def tools_node(state: AgentState) -> dict[str, Any]:
    """Execute every tool call in the latest assistant message."""
    last = state["messages"][-1]
    outputs: list[ToolMessage] = []
    events: list[dict[str, Any]] = []
    updates: dict[str, Any] = {}

    for call in getattr(last, "tool_calls", []) or []:
        name = call["name"]
        args = dict(call.get("args") or {})
        tool = TOOLS_BY_NAME.get(name)

        if tool is None:
            result: dict[str, Any] = {
                "ok": False,
                "error": f"Unknown tool '{name}'. No action was taken.",
            }
        else:
            if name == "create_ticket" and state.get("employee_id"):
                args.setdefault("employee_id", state["employee_id"])
            try:
                result = tool.invoke(args)
            except Exception as exc:  # noqa: BLE001 - defensive boundary
                logger.exception("Tool %s raised", name)
                result = {
                    "ok": False,
                    "error": f"The {name} tool failed: {exc.__class__.__name__}.",
                }

        events.append({"tool": name, "args": args, "result": result})
        outputs.append(
            ToolMessage(
                content=json.dumps(result, default=str),
                name=name,
                tool_call_id=call["id"],
            )
        )

        if name == "get_employee" and result.get("ok"):
            employee = result["employee"]
            updates["employee_profile"] = employee
            updates["employee_id"] = employee["employee_id"]

        if name == "create_ticket" and result.get("ok") and result.get("created"):
            updates["pending_ticket"] = None
            updates["confirmed"] = False

    return {"messages": outputs, "tool_events": events, **updates}


def confirm_node(state: AgentState) -> dict[str, Any]:
    """Hold a ticket creation back until the employee has agreed to it.

    The model asked to write to the ticket system before the employee confirmed.
    The call is answered with an instruction rather than executed, so the graph
    stays valid and the assistant asks for confirmation on its next turn.
    """
    last = state["messages"][-1]
    draft = state.get("pending_ticket")
    outputs: list[ToolMessage] = []

    for call in _write_tool_calls(last):
        draft = _merge_draft(draft, call.get("args") or {}, state.get("employee_id"))
        outputs.append(
            ToolMessage(
                content=CONFIRMATION_REQUEST.format(draft=json.dumps(draft)),
                name=call["name"],
                tool_call_id=call["id"],
            )
        )

    # Any read-only calls made in the same turn are still executed normally.
    for call in getattr(last, "tool_calls", []) or []:
        if call["name"] in WRITE_TOOLS:
            continue
        tool = TOOLS_BY_NAME[call["name"]]
        result = tool.invoke(dict(call.get("args") or {}))
        outputs.append(
            ToolMessage(
                content=json.dumps(result, default=str),
                name=call["name"],
                tool_call_id=call["id"],
            )
        )

    return {
        "messages": outputs,
        "pending_ticket": draft,
        "tool_events": [
            {
                "tool": "confirmation_gate",
                "args": draft or {},
                "result": {
                    "ok": True,
                    "created": False,
                    "note": "Ticket creation paused until the employee confirms.",
                },
            }
        ],
    }


def clarify_node(state: AgentState) -> dict[str, Any]:
    """Ask for the specific details a ticket still needs.

    Reached when the model tried to create a ticket without every required
    field. No ticket is written and the missing field names are handed back.
    """
    last = state["messages"][-1]
    draft = state.get("pending_ticket")
    missing: list[str] = []
    outputs: list[ToolMessage] = []

    for call in _write_tool_calls(last):
        draft = _merge_draft(draft, call.get("args") or {}, state.get("employee_id"))
        missing = missing_ticket_fields(draft)
        outputs.append(
            ToolMessage(
                content=MISSING_INFORMATION.format(fields=", ".join(missing))
                + f"\nValid categories: {', '.join(VALID_CATEGORIES)}.",
                name=call["name"],
                tool_call_id=call["id"],
            )
        )

    return {
        "messages": outputs,
        "pending_ticket": draft,
        "missing_fields": missing,
        "tool_events": [
            {
                "tool": "validation_gate",
                "args": draft or {},
                "result": {
                    "ok": False,
                    "created": False,
                    "error": f"Missing required details: {', '.join(missing)}",
                },
            }
        ],
    }


def respond_node(state: AgentState) -> dict[str, Any]:
    """Finalise the turn.

    The assistant's last message is already the answer whenever the model
    replied with text. This node exists so the workflow always ends at a single,
    explicit response step, and so an empty model reply still yields something
    useful for the employee.
    """
    messages = state.get("messages", [])
    last = messages[-1] if messages else None

    if isinstance(last, AIMessage) and str(last.content).strip():
        return {"route": "respond"}

    return {
        "route": "respond",
        "messages": [
            AIMessage(
                content=(
                    "I was not able to produce an answer for that request. "
                    "Please rephrase it, or tell me your employee ID and the "
                    "problem you are seeing so I can raise a ticket."
                )
            )
        ],
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_agent(state: AgentState) -> str:
    """Conditional edge: decide what happens after the model has spoken."""
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    tool_calls = getattr(last, "tool_calls", None) or []

    if not tool_calls:
        return "respond"

    settings = get_settings()
    if state.get("steps", 0) > settings.max_agent_steps:
        return "respond"

    write_calls = [call for call in tool_calls if call["name"] in WRITE_TOOLS]
    if write_calls:
        draft = state.get("pending_ticket")
        for call in write_calls:
            draft = _merge_draft(draft, call.get("args") or {}, state.get("employee_id"))
        if missing_ticket_fields(draft):
            return "clarify"
        if not state.get("confirmed"):
            return "confirm"

    return "tools"


def route_after_validate(state: AgentState) -> str:
    """Skip the model entirely when validation already answered the user."""
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    if isinstance(last, AIMessage):
        return "respond"
    return "agent"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_graph(model: BaseChatModel | None = None, checkpointer: Any = None):
    """Compile the workflow. Pass a model to inject a stub during testing."""
    model = model if model is not None else build_agent_model(ALL_TOOLS)

    builder = StateGraph(AgentState)
    builder.add_node("validate", validate_node)
    builder.add_node("agent", make_agent_node(model))
    builder.add_node("tools", tools_node)
    builder.add_node("confirm", confirm_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "validate")
    builder.add_conditional_edges(
        "validate", route_after_validate, {"agent": "agent", "respond": "respond"}
    )
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "confirm": "confirm",
            "clarify": "clarify",
            "respond": "respond",
        },
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("confirm", "agent")
    builder.add_edge("clarify", "agent")
    builder.add_edge("respond", END)

    return builder.compile(checkpointer=checkpointer)


def build_checkpointer(settings: Settings | None = None):
    """Create the SQLite checkpointer that persists conversation state."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    settings = settings or get_settings()
    settings.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.checkpoint_file, check_same_thread=False)
    return SqliteSaver(conn)


__all__ = [
    "AgentState",
    "build_checkpointer",
    "build_graph",
    "initial_state",
    "is_affirmative",
    "missing_ticket_fields",
    "route_after_agent",
    "STEP_LIMIT_REACHED",
]
