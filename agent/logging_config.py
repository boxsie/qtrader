"""Process-wide logging setup for the agent.

Mirrors the format used in agent/standby.py so all agent components log the
same way regardless of which entry point boots them.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union

_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
) -> None:
    """Configure the root logger. Idempotent — safe to call from multiple entry points."""
    root = logging.getLogger()
    root.setLevel(level)

    # Drop any handlers that a previous setup_logging() (or basicConfig) installed.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        rot = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=5)
        rot.setFormatter(formatter)
        root.addHandler(rot)


def get_logger(name: str) -> logging.Logger:
    """Thin convenience wrapper so callers can do `logger = get_logger(__name__)`."""
    return logging.getLogger(name)
