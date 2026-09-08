from __future__ import annotations

import os

from config import Settings, settings
from decompose import decompose_query
from generate import generate_answer, get_generator_llm
from hybrid import HybridIndex, build_index, hybrid_search
from rerank import CrossEncoderReranker
from type import RAGResult, RetrievedDoc


class RAGPipeline:
    def __init__(
        self,
        index: HybridIndex | None = None,
        reranker: CrossEncoderReranker | None = None,
        cfg: Settings | None = None,
    ) -> None:
        self.cfg = cfg or settings
        self._configure_langsmith()
        self.index = index or build_index(cfg=self.cfg)
        self.reranker = reranker or CrossEncoderReranker(self.cfg.rerank_model)
        self.llm = get_generator_llm()

    def _configure_langsmith(self) -> None:
        if self.cfg.langchain_tracing_v2 or os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
            project = self.cfg.langchain_project
            os.environ.setdefault("LANGCHAIN_PROJECT", project)
            key = self.cfg.langsmith_api_key or self.cfg.langchain_api_key
            if key:
                os.environ.setdefault("LANGCHAIN_API_KEY", key)
                os.environ.setdefault("LANGSMITH_API_KEY", key)

    def retrieve(self, question: str) -> tuple[list[str], list[RetrievedDoc]]:
        subqueries = decompose_query(question, llm=self.llm)
        merged: dict[str, RetrievedDoc] = {}
        for sub in subqueries:
            for doc in hybrid_search(self.index, sub, cfg=self.cfg):
                prev = merged.get(doc.chunk_id)
                if prev is None or (doc.rrf_score or 0) > (prev.rrf_score or 0):
                    merged[doc.chunk_id] = doc
        candidates = sorted(
            merged.values(),
            key=lambda d: d.rrf_score or 0.0,
            reverse=True,
        )[: self.cfg.hybrid_candidates]
        reranked = self.reranker.rerank(question, candidates, cfg=self.cfg)
        return subqueries, reranked

    def query(self, question: str) -> RAGResult:
        subqueries, contexts = self.retrieve(question)
        answer, generator = generate_answer(question, contexts, llm=self.llm)
        return RAGResult(
            question=question,
            subqueries=subqueries,
            answer=answer,
            contexts=contexts,
            generator=generator,
        )
