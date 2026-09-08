from __future__ import annotations

from config import Settings, settings
from type import RetrievedDoc


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name=model_name or settings.rerank_model)

    def rerank(
        self,
        query: str,
        docs: list[RetrievedDoc],
        top_k: int | None = None,
        cfg: Settings | None = None,
    ) -> list[RetrievedDoc]:
        cfg = cfg or settings
        top_k = top_k or cfg.rerank_top_k
        if not docs:
            return []
        texts = [d.text for d in docs]
        scores = list(self.model.rerank(query, texts))
        ranked = sorted(zip(scores, docs), key=lambda p: float(p[0]), reverse=True)
        out: list[RetrievedDoc] = []
        for score, doc in ranked[:top_k]:
            doc.rerank_score = float(score)
            doc.score = float(score)
            out.append(doc)
        return out
