#!/usr/bin/env python
"""Batch-ingest FAQ documents into the GridTrace knowledge base.

Examples
--------
    uv run python scripts/ingest_docs.py --path data/faq/OpsWarden_FAQ.md
    uv run python scripts/ingest_docs.py --path data/faq/ --recursive
    uv run python scripts/ingest_docs.py --path data/faq/OpsWarden_FAQ.md --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow `python scripts/ingest_docs.py` without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridtrace.loaders import RawEntry, load_entries
from gridtrace.utils.config import get_settings
from gridtrace.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _gather_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        if recursive:
            return sorted(
                p
                for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in {".md", ".markdown", ".json", ".csv"}
            )
        return sorted(
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in {".md", ".markdown", ".json", ".csv"}
        )
    raise FileNotFoundError(f"Path not found: {path}")


def _load(path: Path) -> list[RawEntry]:
    return load_entries(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch-ingest FAQ into GridTrace KB.")
    parser.add_argument(
        "--path", required=True, help="File or directory of .md/.json/.csv FAQ files"
    )
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print, but do not write to storage",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max entries to ingest (per file)"
    )
    args = parser.parse_args(argv)

    configure_logging()
    files = _gather_files(Path(args.path), args.recursive)
    if not files:
        print(f"No input files found under {args.path!r}", file=sys.stderr)
        return 1

    all_entries: list[tuple[Path, RawEntry]] = []
    for f in files:
        entries = _load(f)
        if args.limit:
            entries = entries[: args.limit]
        for e in entries:
            all_entries.append((f, e))

    print(f"Discovered {len(files)} file(s); {len(all_entries)} entries.")

    if args.dry_run:
        for f, e in all_entries[:5]:
            preview = e.solution[:60].replace("\n", " ")
            print(f"  [{f.name}] Q: {e.question}\n           A: {preview}…")
        if len(all_entries) > 5:
            print(f"  … and {len(all_entries) - 5} more")
        return 0

    # real ingest
    from gridtrace.pipeline import GridTracePipeline

    cfg = get_settings()
    pipeline = GridTracePipeline.from_config(cfg)
    try:
        count = 0
        for _f, e in all_entries:
            pipeline.ingest(
                question=e.question,
                solution=e.solution,
                category=e.category,
                doc_id=e.doc_id,
                page_index=e.page_index,
            )
            count += 1
        stats = pipeline.stats()
        print(
            json.dumps(
                {
                    "ingested": count,
                    "anchors": stats.total_anchors,
                    "entries": stats.total_entries,
                    "unique_docs": stats.unique_docs,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
