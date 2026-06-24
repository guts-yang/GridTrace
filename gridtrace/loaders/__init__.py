"""Document loaders.

Each loader returns a list of :class:`RawEntry` records, where each
record carries question / solution / category / doc_id / page_index.
Loaders are intentionally simple: they do not encode, do not call into
storage. The pipeline is responsible for that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

__all__ = ["RawEntry", "load_faq_markdown", "load_faq_json", "load_entries"]


@dataclass(slots=True)
class RawEntry:
    """A pre-encoded entry from a source file."""

    question: str
    solution: str
    category: str | None = None
    doc_id: str = "default"
    page_index: int = 0


def _iter_markdown_faq(text: str, *, doc_id: str) -> Iterable[RawEntry]:
    """Parse a markdown FAQ with the following shape:

    ```
    ## Q1. <question>
    Category: <cat>
    Page: <n>

    <solution paragraphs...>
    ```
    """
    lines = text.splitlines()
    i = 0
    q_index = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("## "):
            q_index += 1
            question = line[3:].strip()
            # strip leading "Qn. " prefix
            if question[:2].lower().startswith("q") and "." in question[:4]:
                question = question.split(".", 1)[1].strip()
            category: str | None = None
            page_index = 0
            # collect metadata lines
            i += 1
            while i < len(lines) and lines[i].strip().startswith(("Category:", "Page:", "Tags:")):
                meta = lines[i].strip()
                if meta.startswith("Category:"):
                    category = meta.split(":", 1)[1].strip() or None
                elif meta.startswith("Page:"):
                    try:
                        page_index = int(meta.split(":", 1)[1].strip())
                    except ValueError:
                        page_index = 0
                i += 1
            # collect solution body (until next "## " or EOF)
            body: list[str] = []
            while i < len(lines) and not lines[i].lstrip().startswith("## "):
                body.append(lines[i].rstrip())
                i += 1
            solution = "\n".join(body).strip()
            if question and solution:
                yield RawEntry(
                    question=question,
                    solution=solution,
                    category=category,
                    doc_id=doc_id,
                    page_index=page_index or q_index,
                )
        else:
            i += 1


def load_faq_markdown(path: str | Path) -> list[RawEntry]:
    """Load entries from a markdown FAQ file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return list(_iter_markdown_faq(text, doc_id=p.stem))


def load_faq_json(path: str | Path) -> list[RawEntry]:
    """Load entries from a JSON file (list of dicts)."""
    import json

    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"JSON FAQ must be a list, got {type(data).__name__}")
    entries: list[RawEntry] = []
    for i, item in enumerate(data, 1):
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        s = str(item.get("solution", item.get("answer", ""))).strip()
        if not q or not s:
            continue
        entries.append(
            RawEntry(
                question=q,
                solution=s,
                category=item.get("category"),
                doc_id=str(item.get("doc_id", p.stem)),
                page_index=int(item.get("page_index", i)),
            )
        )
    return entries


def load_faq_csv(path: str | Path) -> list[RawEntry]:
    """Load entries from a CSV with columns: question, solution [, category, doc_id, page_index]."""
    import csv

    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        entries: list[RawEntry] = []
        for i, row in enumerate(reader, 1):
            q = (row.get("question") or "").strip()
            s = (row.get("solution") or row.get("answer") or "").strip()
            if not q or not s:
                continue
            entries.append(
                RawEntry(
                    question=q,
                    solution=s,
                    category=(row.get("category") or None),
                    doc_id=(row.get("doc_id") or p.stem),
                    page_index=int(row.get("page_index") or i),
                )
            )
    return entries


def load_entries(path: str | Path) -> list[RawEntry]:
    """Auto-detect format by suffix."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return load_faq_markdown(p)
    if suffix == ".json":
        return load_faq_json(p)
    if suffix == ".csv":
        return load_faq_csv(p)
    raise ValueError(f"Unsupported file format: {suffix}")
