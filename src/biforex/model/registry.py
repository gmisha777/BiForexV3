"""Versioned model storage: models/<name>/<version>/{model.txt, meta.json} + current.json.

<name> is a strategy name (e.g. "lgbm_tb"), or "<strategy>/<SYMBOL>" for per-symbol models.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb

from ..config import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"


def save_model(name: str, model: lgb.LGBMClassifier, meta: dict, make_current: bool) -> Path:
    version = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    folder = MODELS_DIR / name / version
    folder.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(folder / "model.txt")
    meta = {"name": name, "version": version, **meta}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    if make_current:
        set_current(name, version)
    return folder


def set_current(name: str, version: str) -> None:
    pointer = MODELS_DIR / name / "current.json"
    pointer.write_text(json.dumps({"version": version}), encoding="utf-8")


def load_current(name: str) -> tuple[lgb.Booster, dict]:
    pointer = MODELS_DIR / name / "current.json"
    version = json.loads(pointer.read_text(encoding="utf-8"))["version"]
    folder = MODELS_DIR / name / version
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    return lgb.Booster(model_file=str(folder / "model.txt")), meta
