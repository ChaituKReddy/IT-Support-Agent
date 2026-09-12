"""Keyword ranking for the knowledge base.

A compact BM25 implementation is used instead of embeddings: the corpus is
small, the results are deterministic, and it introduces no extra dependency or
API cost. Ranking runs over the title, tags and body of each article.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

K1 = 1.5
B = 0.75

# Words that carry no discriminating signal in IT support phrasing.
STOPWORDS = frozenset(
    """
    a an and are as at be but by can do does for from has have how i if in is it
    its me my not of on or that the their there they this to was what when where
    which who will with you your please help need want
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase a string and split it into meaningful word tokens."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Document:
    """One searchable knowledge base article."""

    article_id: str
    title: str
    category: str
    tags: str
    content: str

    def searchable_text(self) -> str:
        # The title and tags are repeated so that matches there outweigh a
        # single incidental mention deep inside the body text.
        return " ".join([self.title, self.title, self.tags, self.tags, self.content])


@dataclass(frozen=True)
class SearchHit:
    """An article together with the score that ranked it."""

    document: Document
    score: float


class BM25Index:
    """In-memory BM25 index over a fixed set of documents."""

    def __init__(self, documents: Sequence[Document]) -> None:
        self.documents = list(documents)
        self._term_frequencies: list[Counter[str]] = []
        self._lengths: list[int] = []
        document_frequency: Counter[str] = Counter()

        for document in self.documents:
            tokens = tokenize(document.searchable_text())
            frequencies = Counter(tokens)
            self._term_frequencies.append(frequencies)
            self._lengths.append(len(tokens))
            document_frequency.update(frequencies.keys())

        self._document_frequency = document_frequency
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )

    def _inverse_document_frequency(self, term: str) -> float:
        total = len(self.documents)
        seen_in = self._document_frequency.get(term, 0)
        if seen_in == 0:
            return 0.0
        return math.log(1 + (total - seen_in + 0.5) / (seen_in + 0.5))

    def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        """Return the highest scoring articles for a free-text query."""
        terms = tokenize(query)
        if not terms or not self.documents:
            return []

        scored: list[SearchHit] = []
        for index, document in enumerate(self.documents):
            frequencies = self._term_frequencies[index]
            length = self._lengths[index] or 1
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                numerator = frequency * (K1 + 1)
                denominator = frequency + K1 * (
                    1 - B + B * length / (self._average_length or 1)
                )
                score += self._inverse_document_frequency(term) * numerator / denominator
            if score > 0:
                scored.append(SearchHit(document=document, score=round(score, 4)))

        scored.sort(key=lambda hit: (-hit.score, hit.document.article_id))
        return scored[:top_k]


def build_index(rows: Iterable[dict]) -> BM25Index:
    """Build an index from database rows or plain dictionaries."""
    return BM25Index(
        [
            Document(
                article_id=row["article_id"],
                title=row["title"],
                category=row["category"],
                tags=row["tags"],
                content=row["content"],
            )
            for row in rows
        ]
    )
