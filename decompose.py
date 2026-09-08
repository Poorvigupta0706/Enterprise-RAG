from __future__ import annotations

import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

_SPLITTERS = [
    r"\s+and also\s+",
    r"\s+as well as\s+",
    r"\s+in addition to\s+",
    r"\?\s+",
    r";\s+",
]

_SYSTEM = (
    "Split a user question into independent search sub-queries. "
    "Return one subquery per line. If the question is already simple, return it unchanged. "
    "Do not answer the question."
)


def decompose_query(question: str, llm: BaseChatModel | None = None) -> list[str]:
    question = question.strip()
    if llm is not None:
        try:
            msg = llm.invoke(
                [
                    SystemMessage(content=_SYSTEM),
                    HumanMessage(content=question),
                ]
            )
            lines = [ln.strip(" -•\t") for ln in str(msg.content).splitlines() if ln.strip()]
            cleaned = [_normalize(x) for x in lines if x]
            if cleaned:
                return _dedupe(cleaned)
        except Exception:
            pass
    return heuristic_decompose(question)


_SPLIT_BEFORE_QUESTION = re.compile(
    r"\s+and\s+(?=(?:what|how|when|where|who|why|which|can the)\b)",
    flags=re.IGNORECASE,
)


def heuristic_decompose(question: str) -> list[str]:
    q = question.strip()
    if not q:
        return []
    parts = [q]
    for pattern in _SPLITTERS:
        next_parts: list[str] = []
        for part in parts:
            next_parts.extend(re.split(pattern, part, flags=re.IGNORECASE))
        parts = [p.strip(" ?") for p in next_parts if p.strip(" ?")]

    expanded: list[str] = []
    for part in parts:
        expanded.extend(p.strip(" ?") for p in _SPLIT_BEFORE_QUESTION.split(part) if p.strip(" ?"))

    result = _dedupe(_normalize(p) for p in expanded if p)
    return result or [q]


def _normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" ?.")
    if text:
        text = text[0].upper() + text[1:]
        if not text.endswith("?"):
            text += "?"
    return text


def _dedupe(items: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
