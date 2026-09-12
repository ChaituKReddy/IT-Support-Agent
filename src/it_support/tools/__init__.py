"""Tool registry exposed to the agent."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from .employee import get_employee
from .knowledge import knowledge_search
from .status import system_status
from .tickets import create_ticket, ticket_lookup

#: Tools that write to the local database and therefore need user confirmation.
WRITE_TOOLS: frozenset[str] = frozenset({"create_ticket"})

ALL_TOOLS: list[BaseTool] = [
    knowledge_search,
    ticket_lookup,
    create_ticket,
    get_employee,
    system_status,
]

TOOLS_BY_NAME: dict[str, BaseTool] = {tool.name: tool for tool in ALL_TOOLS}

__all__ = [
    "ALL_TOOLS",
    "TOOLS_BY_NAME",
    "WRITE_TOOLS",
    "create_ticket",
    "get_employee",
    "knowledge_search",
    "system_status",
    "ticket_lookup",
]
