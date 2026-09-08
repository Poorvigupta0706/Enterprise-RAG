from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    source: str
    text: str


@dataclass
class RetrievedDoc:
    chunk_id: str
    title: str
    source: str
    text: str
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


@dataclass
class RAGResult:
    question: str
    subqueries: list[str]
    answer: str
    contexts: list[RetrievedDoc]
    generator: str
