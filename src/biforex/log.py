"""Logging setup: console + rotating file."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(FORMAT, "%H:%M:%S"))
    root.addHandler(console)

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        file = RotatingFileHandler(
            log_dir / "biforex.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8"
        )
        file.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(file)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
