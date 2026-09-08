from __future__ import annotations

import argparse
import json
import sys

from rag_pipeline.pipeline import RAGPipeline


def query_main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Query the hybrid RAG pipeline")
    parser.add_argument("question", nargs="+")
    args = parser.parse_args()
    question = " ".join(args.question)
    rag = RAGPipeline()
    result = rag.query(question)
    print(
        json.dumps(
            {
                "question": result.question,
                "subqueries": result.subqueries,
                "generator": result.generator,
                "answer": result.answer,
                "contexts": [
                    {
                        "id": c.chunk_id,
                        "title": c.title,
                        "rerank_score": c.rerank_score,
                        "rrf_score": c.rrf_score,
                    }
                    for c in result.contexts
                ],
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    query_main()
