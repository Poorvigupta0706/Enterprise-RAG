from __future__ import annotations

from type import RetrievedDoc


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedDoc]],
    k: int = 60,
) -> list[RetrievedDoc]:
    """Merge ranked lists with Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    docs: dict[str, RetrievedDoc] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            scores[doc.chunk_id] = scores.get(doc.chunk_id, 0.0) + 1.0 / (k + rank)
            existing = docs.get(doc.chunk_id)
            if existing is None:
                docs[doc.chunk_id] = doc
            else:
                if doc.dense_rank is not None:
                    existing.dense_rank = doc.dense_rank
                if doc.bm25_rank is not None:
                    existing.bm25_rank = doc.bm25_rank

    fused: list[RetrievedDoc] = []
    for chunk_id, rrf in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        doc = docs[chunk_id]
        doc.rrf_score = rrf
        doc.score = rrf
        fused.append(doc)
    return fused
