"""Decision strategies and their registry.

To add a strategy: create a module here with a Strategy subclass decorated with @register,
import it at the bottom of this file, and add it to `strategies:` in config.yaml.
"""

from __future__ import annotations

import pandas as pd

from ..config import PROJECT_ROOT, Settings
from .base import Strategy, WalkForwardScores, clip_score

REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    if cls.name in REGISTRY:
        raise ValueError(f"strategy {cls.name!r} registered twice")
    REGISTRY[cls.name] = cls
    return cls


def create(name: str, settings: Settings) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; available: {sorted(REGISTRY)}")
    cfg = settings.strategies.get(name)
    return REGISTRY[name](settings, cfg.params if cfg else {})


def oos_path(strategy: Strategy | str):
    name = strategy if isinstance(strategy, str) else strategy.name
    return PROJECT_ROOT / "reports" / name / "oos_scores.parquet"


def save_oos_scores(strategy: Strategy, scores: pd.DataFrame) -> None:
    path = oos_path(strategy)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores[["symbol", "score"]].to_parquet(path)


def load_oos_scores(name: str) -> pd.DataFrame:
    path = oos_path(name)
    if not path.exists():
        raise FileNotFoundError(f"no walk-forward scores for {name}: run scripts/train.py "
                                f"--strategies {name}")
    return pd.read_parquet(path)


# strategy modules register themselves on import
from . import lgbm_tb  # noqa: E402, F401

__all__ = ["REGISTRY", "Strategy", "WalkForwardScores", "clip_score", "create", "register",
           "load_oos_scores", "save_oos_scores", "oos_path"]
