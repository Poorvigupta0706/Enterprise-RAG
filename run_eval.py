from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from metrics import mean, passes_thresholds, score_sample
from pipeline import RAGPipeline
def load_dataset(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))
def load_thresholds(path: Path) -> dict[str, float]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(k): float(v) for k, v in raw.items()}
def run_pipeline_eval(dataset: list[dict]) -> dict:
    rag = RAGPipeline()
    rows = []
    for item in dataset:
        result = rag.query(item["question"])
        scores = score_sample(
            question=item["question"],
            answer=result.answer,
            ground_truth=item["ground_truth"],
            contexts=result.contexts,
        )
        context_texts = [c.text for c in result.contexts]
        rows.append(
            {
                "question": item["question"],
                "complex": bool(item.get("complex")),
                "subqueries": result.subqueries,
                "generator": result.generator,
                "answer": result.answer,
                "ground_truth": item["ground_truth"],
                "context_texts": context_texts,
                "retrieved": [
                    {
                        "id": c.chunk_id,
                        "title": c.title,
                        "rerank_score": c.rerank_score,
                        "rrf_score": c.rrf_score,
                    }
                    for c in result.contexts
                ],
                "scores": scores,
            }
        )
    averages = {
        "faithfulness": round(mean([r["scores"]["faithfulness"] for r in rows]), 4),
        "answer_relevancy": round(mean([r["scores"]["answer_relevancy"] for r in rows]), 4),
        "context_precision": round(mean([r["scores"]["context_precision"] for r in rows]), 4),
    }
    ragas_block = try_official_ragas(rows)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": "ragas-local-metrics",
        "averages": averages,
        "ragas_official": ragas_block,
        "n_samples": len(rows),
        "samples": rows,
    }
def try_official_ragas(rows: list[dict]) -> dict:
    """Run library RAGAS judges when an OpenAI key is present."""
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return {"status": "skipped", "reason": "OPENAI_API_KEY not set"}
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, ResponseRelevancy
    except Exception as exc:  # pragma: no cover - optional path
        return {"status": "skipped", "reason": f"ragas import failed: {exc}"}
    try:
        samples = [
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r.get("context_texts") or [],
                reference=r["ground_truth"],
            )
            for r in rows
        ]
        llm = LangchainLLMWrapper(ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0))
        emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
        metrics = [
            Faithfulness(llm=llm),
            ResponseRelevancy(llm=llm, embeddings=emb),
            LLMContextPrecisionWithReference(llm=llm),
        ]
        dataset = EvaluationDataset(samples=samples)
        result = evaluate(dataset=dataset, metrics=metrics)
        scores = dict(result)
        return {"status": "ok", "scores": scores}
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}
def print_report(report: dict, thresholds: dict[str, float]) -> bool:
    av = report["averages"]
    print("\n=== RAGAS eval suite (faithfulness / relevancy / context precision) ===")
    print(f"samples: {report['n_samples']}")
    print(f"backend: {report['backend']}")
    for name in ("faithfulness", "answer_relevancy", "context_precision"):
        floor = thresholds.get(name)
        extra = f"  (threshold {floor:.2f})" if floor is not None else ""
        print(f"  {name:20s} {av[name]:.4f}{extra}")
    official = report.get("ragas_official") or {}
    print(f"official ragas: {official.get('status')} {official.get('reason', official.get('scores', ''))}")
    ok, failures = passes_thresholds(av, thresholds)
    if ok:
        print("thresholds: PASS")
    else:
        print("thresholds: FAIL")
        for line in failures:
            print(f"  - {line}")
    print("\nPer-question scores:")
    for row in report["samples"]:
        s = row["scores"]
        flag = "complex" if row["complex"] else "simple "
        print(
            f"  [{flag}] F={s['faithfulness']:.2f} R={s['answer_relevancy']:.2f} "
            f"P={s['context_precision']:.2f}  {row['question'][:72]}"
        )
        print(f"          subqueries: {row['subqueries']}")
    return ok
def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid RAG + RAGAS evaluation suite")
    parser.add_argument("--dataset", default=str(ROOT / "evals" / "dataset.json"))
    parser.add_argument("--thresholds", default=str(ROOT / "evals" / "threshols.yaml"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "ragas_report.json"))
    args = parser.parse_args()
    dataset = load_dataset(Path(args.dataset))
    thresholds = load_thresholds(Path(args.thresholds))
    report = run_pipeline_eval(dataset)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    ok = print_report(report, thresholds)
    print(f"\nWrote {out}")
    raise SystemExit(0 if ok else 1)
if __name__ == "__main__":
    main()