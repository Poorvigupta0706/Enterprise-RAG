from __future__ import annotations

from pathlib import Path

from type import Chunk

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "data" / "corpus"


def load_chunks(corpus_dir: Path | None = None) -> list[Chunk]:
    base = corpus_dir or CORPUS_DIR
    chunks: list[Chunk] = []
    for path in sorted(base.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        sections = _split_markdown(text, path.stem)
        chunks.extend(sections)
    if not chunks:
        raise FileNotFoundError(f"No markdown corpus found in {base}")
    return chunks


def _split_markdown(text: str, source: str) -> list[Chunk]:
    parts: list[Chunk] = []
    current_title = source.replace("-", " ").title()
    buf: list[str] = []
    idx = 0

    def flush() -> None:
        nonlocal idx
        body = "\n".join(buf).strip()
        if not body:
            return
        chunk_id = f"{source}-{idx:03d}"
        parts.append(Chunk(id=chunk_id, title=current_title, source=source, text=body))
        idx += 1
        buf.clear()

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            current_title = line[3:].strip()
            continue
        buf.append(line)
    flush()
    return parts
