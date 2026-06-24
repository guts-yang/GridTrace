#!/usr/bin/env python
"""Destructively reset the GridTrace knowledge base.

This is intended for development and CI. Refuses to run without
``--confirm`` (unless ``--dry-run``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridtrace.utils.config import get_settings
from gridtrace.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reset the GridTrace knowledge base.")
    parser.add_argument("--confirm", action="store_true", help="Actually delete data")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would happen, then exit"
    )
    args = parser.parse_args(argv)

    configure_logging()
    if not args.confirm and not args.dry_run:
        print("Refusing to reset without --confirm (use --dry-run to preview).", file=sys.stderr)
        return 2

    cfg = get_settings()
    print(f"Backend: {cfg.storage_backend}")
    if cfg.storage_backend == "postgres":
        print(f"DATABASE_URL: {cfg.database_url}")
    elif cfg.storage_backend == "sqlite":
        print(f"SQLite path:  {cfg.sqlite_path}")

    if args.dry_run:
        print("DRY-RUN: no changes made.")
        return 0

    from gridtrace.pipeline import GridTracePipeline

    pipeline = GridTracePipeline.from_config(cfg)
    try:
        before = pipeline.stats()
        print(
            f"Before: anchors={before.total_anchors} entries={before.total_entries} docs={before.unique_docs}"
        )
        pipeline.storage.reset()
        after = pipeline.stats()
        print(
            f"After:  anchors={after.total_anchors} entries={after.total_entries} docs={after.unique_docs}"
        )
    finally:
        pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
