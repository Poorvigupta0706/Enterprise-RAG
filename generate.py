from __future__ import annotations
import os
import re
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from config import settings
from type import RetrievedDoc
_SYSTEM = (
 """
          You are Nimbus Cloud support.

Use ONLY information present in the provided context.

Do NOT use prior knowledge.
Do NOT make calculations.
Do NOT add explanations that are not explicitly written in the context.
Do NOT infer missing facts.

If the answer is not explicitly present in the context, respond exactly:

I do not know.

Answer in 1-3 sentences maximum.
"""
)
def get_generator_llm() -> BaseChatModel:
    return ChatOllama(
        model="qwen2.5-coder:7b",
        temperature=0
    )
def generate_answer(
    question: str,
    contexts: list[RetrievedDoc],
    llm: BaseChatModel | None = None,
) -> tuple[str, str]:
    packed = _pack_context(contexts)
    print("\n===== RETRIEVED CONTEXT =====")
    print(packed)
    print("=============================\n")
    if llm is None:
        llm = get_generator_llm()
    if llm is not None:
        msg = llm.invoke(
            [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=f"Context:\n{packed}\n\nQuestion: {question}"),
            ]
        )
        return str(msg.content).strip(), "qwen2.5-coder:7b"
    return extractive_answer(question, contexts), "extractive"
def extractive_answer(question: str, contexts: list[RetrievedDoc]) -> str:
    if not contexts:
        return "I do not know."
    q_tokens = _tokens(question)
    picked: list[str] = []
    seen: set[str] = set()
    for doc in contexts[:3]:
        best: tuple[int, str] | None = None
        for sent in _sentences(doc.text):
            overlap = len(q_tokens & _tokens(sent))
            if best is None or overlap > best[0]:
                best = (overlap, sent)
        if best and best[1] not in seen:
            seen.add(best[1])
            picked.append(f"{best[1]} ({doc.title})")
        if len(picked) >= 2:
            break
    if not picked:
        return contexts[0].text[:400]
    return " ".join(picked)
def _pack_context(contexts: list[RetrievedDoc]) -> str:
    blocks = []
    for i, doc in enumerate(contexts, start=1):
        blocks.append(f"[{i}] {doc.title} ({doc.source})\n{doc.text}")
    return "\n\n".join(blocks)
def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]
def _tokens(text: str) -> set[str]:
    stop = {
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
        "what",
        "how",
        "when",
        "where",
        "who",
        "do",
        "does",
        "i",
        "my",
        "we",
    }
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in stop and len(t) > 2}