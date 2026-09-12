"""Knowledge base search tool (Tool 1 of the required three)."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from .. import db
from ..config import get_settings
from ..search import BM25Index, build_index
from .base import fail, ok, safe_tool

_INDEX_CACHE: dict[str, BM25Index] = {}

# Below this BM25 score a "match" is a coincidental word overlap rather than a
# genuinely relevant article, so it is discarded instead of shown to the user.
MIN_RELEVANCE_SCORE = 0.8


def get_index(refresh: bool = False) -> BM25Index:
    """Return the cached knowledge base index, building it on first use."""
    settings = get_settings()
    key = str(settings.database_file)
    if refresh:
        _INDEX_CACHE.pop(key, None)
    if key not in _INDEX_CACHE:
        with db.connection(settings) as conn:
            rows = [dict(r) for r in conn.execute("SELECT * FROM kb_articles")]
        _INDEX_CACHE[key] = build_index(rows)
    return _INDEX_CACHE[key]


def clear_index_cache() -> None:
    """Drop cached indexes. Used by the test-suite between temporary databases."""
    _INDEX_CACHE.clear()


@safe_tool
def _search(query: str, top_k: int | None = None) -> dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return fail("A search query is required to look up knowledge base articles.")

    settings = get_settings()
    limit = top_k or settings.kb_search_top_k
    hits = [
        hit
        for hit in get_index().search(query, top_k=limit)
        if hit.score >= MIN_RELEVANCE_SCORE
    ]

    if not hits:
        return ok(
            query=query,
            articles=[],
            note=(
                "No knowledge base article matched this query. Do not invent "
                "guidance; offer to raise a ticket instead."
            ),
        )

    return ok(
        query=query,
        articles=[
            {
                "article_id": hit.document.article_id,
                "title": hit.document.title,
                "category": hit.document.category,
                "content": hit.document.content,
                "relevance": hit.score,
            }
            for hit in hits
        ],
    )


@tool("knowledge_search")
def knowledge_search(query: str) -> dict[str, Any]:
    """Search the internal IT knowledge base for how-to and troubleshooting articles.

    Use this for any question about how to do something or why something is not
    working, for example resetting a VPN password, fixing Outlook, or printing
    problems. Returns the most relevant articles with their full text.

    Args:
        query: What the employee wants to know, in their own words.
    """
    return _search(query)
