"""Versioned model storage: models/<SYMBOL>/<version>/{model.txt, meta.json} + current.json."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb

from ..config import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"
POOLED = "POOLED"  # name of the model shared by all symbols


def save_model(symbol: str, model: lgb.LGBMClassifier, meta: dict, make_current: bool) -> Path:
    version = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    folder = MODELS_DIR / symbol / version
    folder.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(folder / "model.txt")
    meta = {"symbol": symbol, "version": version, **meta}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    if make_current:
        set_current(symbol, version)
    return folder


def set_current(symbol: str, version: str) -> None:
    pointer = MODELS_DIR / symbol / "current.json"
    pointer.write_text(json.dumps({"version": version}), encoding="utf-8")


def load_current(symbol: str) -> tuple[lgb.Booster, dict]:
    pointer = MODELS_DIR / symbol / "current.json"
    version = json.loads(pointer.read_text(encoding="utf-8"))["version"]
    folder = MODELS_DIR / symbol / version
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    return lgb.Booster(model_file=str(folder / "model.txt")), meta
