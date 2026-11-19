"""Central logging setup."""
from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=_FMT, handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
