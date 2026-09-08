from __future__ import annotations
import re
from collections.abc import Sequence
from type import RetrievedDoc
_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "for",
    "to",
    "in",
    "on",
    "is",
    "are",
    "was",
    "be",
    "as",
    "with",
    "that",
    "this",
    "it",
    "what",
    "how",
    "when",
    "where",
    "who",
    "why",
    "which",
    "does",
    "do",
    "did",
    "can",
    "could",
    "i",
    "my",
    "we",
    "there",
}
def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1}
def token_f1(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb)
    if overlap == 0:
        return 0.0
    precision = overlap / len(ta)
    recall = overlap / len(tb)
    return 2 * precision * recall / (precision + recall)
def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 12]
def faithfulness(answer: str, contexts: Sequence[str]) -> float:
    """Fraction of answer sentences supported by retrieved context (RAGAS-style)."""
    claims = sentences(answer)
    if not claims:
        return 0.0
    blob = " ".join(contexts)
    supported = sum(1 for c in claims if token_f1(c, blob) >= 0.25 or _contained(c, blob))
    return supported / len(claims)
def answer_relevancy(question: str, answer: str) -> float:
    """Share of question content terms that appear in the answer (RAGAS-style proxy)."""
    if not answer or answer.lower().startswith("i do not know"):
        return 0.0
    tq, ta = tokens(question), tokens(answer)
    if not tq or not ta:
        return 0.0
    coverage = len(tq & ta) / len(tq)
    return max(0.0, min(1.0, coverage))
def context_precision(ground_truth: str, contexts: Sequence[str]) -> float:
    """Average precision of relevant retrieved chunks vs the reference answer."""
    if not contexts:
        return 0.0
    relevancy = [token_f1(ctx, ground_truth) >= 0.18 or _contained_overlap(ctx, ground_truth) for ctx in contexts]
    hits = 0
    ap = 0.0
    for i, rel in enumerate(relevancy, start=1):
        if rel:
            hits += 1
            ap += hits / i
    return ap / hits if hits else 0.0
def score_sample(
    question: str,
    answer: str,
    ground_truth: str,
    contexts: Sequence[str] | Sequence[RetrievedDoc],
) -> dict[str, float]:
    texts = [c.text if isinstance(c, RetrievedDoc) else c for c in contexts]
    return {
        "faithfulness": round(faithfulness(answer, texts), 4),
        "answer_relevancy": round(answer_relevancy(question, answer), 4),
        "context_precision": round(context_precision(ground_truth, texts), 4),
    }
def _contained(claim: str, blob: str) -> bool:
    ct = tokens(claim)
    bt = tokens(blob)
    if len(ct) < 4:
        return ct <= bt
    return len(ct & bt) / len(ct) >= 0.6
def _contained_overlap(ctx: str, truth: str) -> bool:
    ct, tt = tokens(ctx), tokens(truth)
    if not tt:
        return False
    return len(ct & tt) / len(tt) >= 0.45
def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
def passes_thresholds(averages: dict[str, float], thresholds: dict[str, float]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for name, floor in thresholds.items():
        got = averages.get(name, 0.0)
        if got + 1e-9 < float(floor):
            failures.append(f"{name}: {got:.3f} < {float(floor):.3f}")
    return (not failures, failures)