"""Streamlit chat interface for the IT Support Assistant.

Left pane: the conversation, with the identified employee highlighted and every
tool call shown. Right pane: the local data the assistant works from, so an
evaluator can browse the knowledge base and the ticket database directly.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import json
import logging
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from it_support import browse, db  # noqa: E402
from it_support.config import get_settings  # noqa: E402
from it_support.graph import build_checkpointer, build_graph  # noqa: E402
from it_support.llm import LLMConfigurationError, describe_provider  # noqa: E402
from it_support.ui_theme import (  # noqa: E402
    STYLESHEET,
    priority_chip,
    service_chip,
    status_chip,
)

SAMPLE_PROMPTS = [
    "How do I reset my VPN password?",
    "What is the status of my laptop issue? I am EMP1115.",
    "My VPN is not working. Please raise a ticket. I am EMP1024.",
    "Is the print service down?",
]

EMPLOYEE_ID_PATTERN = re.compile(r"\b(EMP\d{4})\b")
TICKET_ID_PATTERN = re.compile(r"\b(INC-\d{5})\b")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def configure_logging() -> None:
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@st.cache_resource(show_spinner=False)
def load_assistant():
    """Compile the workflow once per server process."""
    settings = get_settings()
    db.initialise(settings)
    return build_graph(checkpointer=build_checkpointer(settings))


def new_thread_id() -> str:
    return f"session-{uuid.uuid4().hex[:12]}"


def reset_conversation() -> None:
    st.session_state.transcript = []
    st.session_state.thread_id = new_thread_id()
    st.session_state.identity = None


def queue_prompt(prompt: str) -> None:
    st.session_state.queued_prompt = prompt


# ---------------------------------------------------------------------------
# Conversation pane
# ---------------------------------------------------------------------------


def highlight_identifiers(text: str) -> str:
    """Mark employee and ticket references so they stand out in an answer."""
    marked = EMPLOYEE_ID_PATTERN.sub(r'<span class="empid-inline">\1</span>', text)
    return TICKET_ID_PATTERN.sub(r'<span class="empid-inline">\1</span>', marked)


def render_identity(profile: dict[str, Any] | None, employee_id: str | None) -> None:
    """Banner showing who the assistant currently believes it is helping."""
    if profile:
        initials = "".join(part[0] for part in profile["name"].split()[:2]).upper()
        st.markdown(
            f"""<div class="identity">
              <div class="avatar">{html.escape(initials)}</div>
              <div>
                <div class="who">{html.escape(profile['name'])}
                  <span class="empid">{html.escape(profile['employee_id'])}</span>
                </div>
                <div class="meta">{html.escape(profile['department'])} ·
                  {html.escape(profile['location'])} ·
                  VPN {'enabled' if profile['vpn_enabled'] else 'not enabled'} ·
                  MFA {'enrolled' if profile['mfa_enrolled'] else 'not enrolled'}</div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
    elif employee_id:
        st.markdown(
            f"""<div class="identity">
              <div class="avatar">ID</div>
              <div>
                <div class="who">Employee
                  <span class="empid">{html.escape(employee_id)}</span>
                </div>
                <div class="meta">Identifier captured from the conversation.
                  The directory has not been queried yet.</div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div class="identity anonymous">
              <div class="avatar">?</div>
              <div>
                <div class="who">No employee identified yet</div>
                <div class="meta">Mention your employee ID (for example EMP1024)
                  to check tickets or raise one.</div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_tool_event(event: dict[str, Any]) -> None:
    """Show one tool call and what the local system actually returned."""
    result = event.get("result", {})
    icon = "✅" if result.get("ok") else "⚠️"
    with st.expander(f"{icon} `{event['tool']}`", expanded=False):
        st.caption("Arguments")
        st.code(json.dumps(event.get("args", {}), indent=2), language="json")
        st.caption("Result from the local system")
        st.code(json.dumps(result, indent=2, default=str), language="json")


def tickets_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tickets: list[dict[str, Any]] = []
    for event in events:
        result = event.get("result", {})
        if event["tool"] == "ticket_lookup" and result.get("tickets"):
            tickets.extend(result["tickets"])
        if event["tool"] == "create_ticket":
            for key in ("ticket", "duplicate_of"):
                if result.get(key):
                    tickets.append(result[key])
    seen: set[str] = set()
    unique = []
    for ticket in tickets:
        if ticket["ticket_id"] not in seen:
            seen.add(ticket["ticket_id"])
            unique.append(ticket)
    return unique


def render_ticket_cards(tickets: list[dict[str, Any]]) -> None:
    for ticket in tickets:
        st.markdown(
            f"""<div class="ticket-card">
              <div class="row">
                <span class="tid">{html.escape(ticket['ticket_id'])}</span>
                <span>{status_chip(ticket['status'])} {priority_chip(ticket['priority'])}</span>
              </div>
              <div class="subject">{html.escape(ticket['subject'])}</div>
              <div class="meta">{html.escape(ticket['category'])} ·
                {html.escape(ticket['employee_id'])} ·
                {html.escape(ticket.get('assigned_to') or 'Unassigned')} ·
                updated {html.escape(ticket['updated_at'][:10])}</div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_turn(turn: dict[str, Any]) -> None:
    """Render one stored assistant turn: answer, tickets touched, tool trace."""
    st.markdown(highlight_identifiers(turn["answer"]), unsafe_allow_html=True)

    tickets = tickets_from_events(turn["tool_events"])
    if tickets:
        render_ticket_cards(tickets)

    if turn["tool_events"]:
        names = ", ".join(f"`{event['tool']}`" for event in turn["tool_events"])
        st.caption(f"Tools used: {names}")
        for event in turn["tool_events"]:
            render_tool_event(event)


# ---------------------------------------------------------------------------
# Reference pane
# ---------------------------------------------------------------------------


def render_knowledge_panel() -> None:
    st.markdown('<div class="panel-title">Knowledge base</div>', unsafe_allow_html=True)
    query = st.text_input(
        "Filter articles",
        key="kb_filter",
        placeholder="vpn password, printing, outlook…",
        label_visibility="collapsed",
    )
    articles = browse.list_articles(query)
    st.caption(f"{len(articles)} of 15 articles")

    for article in articles:
        relevance = article.get("relevance")
        suffix = f" · relevance {relevance:.2f}" if relevance else ""
        st.markdown(
            f"""<div class="kb-card">
              <div class="kb-title">{html.escape(article['title'])}</div>
              <div class="kb-meta">{html.escape(article['article_id'])} ·
                {html.escape(article['category'])}{suffix}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        with st.expander("Read article", expanded=False):
            st.write(article["content"])
            st.caption(f"Tags: {article['tags']}")
            st.button(
                "Ask the assistant about this",
                key=f"ask-{article['article_id']}",
                width="stretch",
                on_click=queue_prompt,
                args=(article["title"],),
            )


def render_ticket_panel() -> None:
    st.markdown(
        '<div class="panel-title">Ticket database</div>', unsafe_allow_html=True
    )
    counts = browse.ticket_counts()
    st.markdown(
        '<div class="counters">'
        + "".join(
            f'<div class="counter"><div class="n">{count}</div>'
            f'<div class="l">{html.escape(status)}</div></div>'
            for status, count in counts.items()
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    filters = st.columns(2)
    status = filters[0].selectbox("Status", ["All", *browse.TICKET_STATUSES])
    category = filters[1].selectbox("Category", ["All", *browse.ticket_categories()])

    tickets = browse.list_tickets(
        status="" if status == "All" else status,
        category="" if category == "All" else category,
    )
    st.caption(f"{len(tickets)} tickets")
    st.dataframe(
        [
            {
                "Ticket": t["ticket_id"],
                "Status": t["status"],
                "Priority": t["priority"],
                "Category": t["category"],
                "Employee": t["employee_id"],
                "Subject": t["subject"],
                "Assigned to": t["assigned_to"],
                "Created": t["created_at"][:10],
            }
            for t in tickets
        ],
        hide_index=True,
        width="stretch",
        height=430,
    )


def render_directory_panel() -> None:
    st.markdown(
        '<div class="panel-title">Employee directory</div>', unsafe_allow_html=True
    )
    st.dataframe(
        [
            {
                "ID": e["employee_id"],
                "Name": e["name"],
                "Department": e["department"],
                "Location": e["location"],
                "VPN": "yes" if e["vpn_enabled"] else "no",
                "MFA": "yes" if e["mfa_enrolled"] else "no",
            }
            for e in browse.list_employees()
        ],
        hide_index=True,
        width="stretch",
        height=300,
    )

    st.markdown(
        '<div class="panel-title">Service status</div>', unsafe_allow_html=True
    )
    for service in browse.service_status():
        st.markdown(
            f"""<div class="kb-card">
              <div class="row" style="display:flex;justify-content:space-between;
                   align-items:center;gap:.5rem">
                <span class="kb-title">{html.escape(service['service'])}</span>
                {service_chip(service['status'])}
              </div>
              <div class="kb-meta">{html.escape(service['region'])} ·
                {html.escape(service['message'])}</div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_reference_pane() -> None:
    # Anchor the CSS uses to turn this column into a docked right rail.
    st.markdown('<div class="rail-anchor"></div>', unsafe_allow_html=True)
    knowledge, tickets, directory = st.tabs(
        ["📚 Knowledge base", "🎫 Tickets", "👥 Directory & status"]
    )
    with knowledge:
        render_knowledge_panel()
    with tickets:
        render_ticket_panel()
    with directory:
        render_directory_panel()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def sidebar(settings) -> None:
    with st.sidebar:
        st.subheader("Session")
        st.caption(f"Model · {describe_provider(settings)}")
        st.caption(f"Thread · `{st.session_state.thread_id}`")
        st.button(
            "🧹 Clear conversation",
            width="stretch",
            on_click=reset_conversation,
        )

        st.divider()
        st.subheader("Try asking")
        for prompt in SAMPLE_PROMPTS:
            st.button(
                prompt,
                width="stretch",
                key=f"sample-{prompt}",
                on_click=queue_prompt,
                args=(prompt,),
            )

        st.divider()
        st.subheader("Tools")
        st.markdown(
            "- `knowledge_search` — how-to articles\n"
            "- `ticket_lookup` — existing tickets\n"
            "- `create_ticket` — raises a ticket, asks you first\n"
            "- `get_employee` — employee directory\n"
            "- `system_status` — service health"
        )

        st.divider()
        st.caption(
            "All records are local sample data. The assistant reports only what "
            "these sources return; anything else is labelled as a suggestion."
        )


# ---------------------------------------------------------------------------
# Turn handling
# ---------------------------------------------------------------------------


def answer(assistant, prompt: str) -> dict[str, Any]:
    """Run one turn of the workflow and pull out what the UI needs."""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    before = len((assistant.get_state(config).values or {}).get("tool_events", []))
    result = assistant.invoke({"messages": [("user", prompt)]}, config=config)

    st.session_state.identity = {
        "employee_id": result.get("employee_id"),
        "profile": result.get("employee_profile"),
    }
    return {
        "answer": str(result["messages"][-1].content).strip()
        or "I could not produce an answer for that request.",
        "tool_events": result.get("tool_events", [])[before:],
    }


def render_conversation(assistant, prompt: str | None) -> None:
    """Render the transcript, then handle the prompt taken from the docked input."""
    identity = st.session_state.identity or {}
    render_identity(identity.get("profile"), identity.get("employee_id"))

    for turn in st.session_state.transcript:
        with st.chat_message("user"):
            st.markdown(highlight_identifiers(turn["question"]), unsafe_allow_html=True)
        with st.chat_message("assistant"):
            render_turn(turn)

    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(highlight_identifiers(prompt), unsafe_allow_html=True)

    with st.chat_message("assistant"):
        with st.spinner("Checking the local systems…"):
            try:
                turn = answer(assistant, prompt)
            except Exception as exc:  # noqa: BLE001 - never break the chat loop
                logging.exception("Turn failed")
                turn = {
                    "answer": (
                        "Something went wrong while handling that request "
                        f"({exc.__class__.__name__}). Please try again, or clear "
                        "the conversation if the problem continues."
                    ),
                    "tool_events": [],
                }
        turn["question"] = prompt
        render_turn(turn)

    st.session_state.transcript.append(turn)
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="IT Support Assistant", page_icon="🛠️", layout="wide"
    )
    configure_logging()
    settings = get_settings()

    st.session_state.setdefault("transcript", [])
    st.session_state.setdefault("thread_id", new_thread_id())
    st.session_state.setdefault("queued_prompt", None)
    st.session_state.setdefault("identity", None)

    st.markdown(STYLESHEET, unsafe_allow_html=True)
    st.markdown(
        """<div class="app-header">
          <div class="glyph">🛠️</div>
          <div>
            <h1>IT Support Assistant</h1>
            <p>Agentic help desk for Sourcify · knowledge search, ticket lookup
               and ticket creation over local data</p>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    sidebar(settings)

    try:
        assistant = load_assistant()
    except LLMConfigurationError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # noqa: BLE001 - startup failures must be visible
        st.error(f"The assistant could not start: {exc}")
        st.stop()

    # Declared at page level so Streamlit docks it at the bottom of the window
    # instead of dropping it inline under the identity banner.
    prompt = st.chat_input("Describe your IT issue…")
    if not prompt and st.session_state.queued_prompt:
        prompt = st.session_state.queued_prompt
        st.session_state.queued_prompt = None

    conversation, reference = st.columns([2.6, 1], gap="large")
    with conversation:
        render_conversation(assistant, prompt)
    with reference:
        render_reference_pane()


if __name__ == "__main__":
    main()
