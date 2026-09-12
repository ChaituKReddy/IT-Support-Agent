"""BM25 ranking behaviour."""

from __future__ import annotations

from it_support.search import build_index, tokenize

DOCUMENTS = [
    {
        "article_id": "A",
        "title": "Reset your VPN password",
        "category": "VPN",
        "tags": "vpn, password, reset",
        "content": "Open the portal and reset the VPN credentials.",
    },
    {
        "article_id": "B",
        "title": "Printing problems",
        "category": "Printer",
        "tags": "printer, queue",
        "content": "Restart the print spooler to clear a stuck queue.",
    },
]


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("How do I reset my VPN password?") == ["reset", "vpn", "password"]


def test_search_ranks_the_relevant_article_first():
    hits = build_index(DOCUMENTS).search("reset vpn password", top_k=2)
    assert hits[0].document.article_id == "A"
    assert hits[0].score > 0


def test_search_returns_nothing_for_an_unrelated_query():
    assert build_index(DOCUMENTS).search("payroll bonus calculation") == []


def test_empty_query_returns_no_hits():
    assert build_index(DOCUMENTS).search("   ") == []
