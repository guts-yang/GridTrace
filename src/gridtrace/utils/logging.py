"""Structured logging configuration for GridTrace."""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

_LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)


def configure_logging(level: str | int | None = None) -> None:
    """Configure root logger once. Safe to call multiple times."""
    if isinstance(level, str):
        level_int = logging.getLevelName(level.upper())
        if not isinstance(level_int, int):
            level_int = logging.INFO
    elif isinstance(level, int):
        level_int = level
    else:
        env_level = os.getenv("LOG_LEVEL", "INFO")
        level_int = logging.getLevelName(env_level.upper())
        if not isinstance(level_int, int):
            level_int = logging.INFO

    root = logging.getLogger()
    if root.handlers:        # already configured
        root.setLevel(level_int)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(level_int)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring logging on first call."""
    configure_logging()
    return logging.getLogger(name)
