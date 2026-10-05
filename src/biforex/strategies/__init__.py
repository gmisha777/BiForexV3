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


def oos_path(name: str, report_key: str):
    """report_key: settings.report_key (the timeframe, plus a research tag if any)."""
    return PROJECT_ROOT / "reports" / name / report_key / "oos_scores.parquet"


def save_oos_scores(strategy: Strategy, scores: pd.DataFrame) -> None:
    path = oos_path(strategy.name, strategy.settings.report_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores[["symbol", "score"]].to_parquet(path)


def load_oos_scores(name: str, report_key: str) -> pd.DataFrame:
    path = oos_path(name, report_key)
    if not path.exists():
        raise FileNotFoundError(f"no {report_key} walk-forward scores for {name}: run "
                                f"scripts/train.py --strategies {name}")
    return pd.read_parquet(path)


# strategy modules register themselves on import
from . import (  # noqa: E402, F401
    crowd,
    flow_1m,
    lgbm_m1nn,
    lgbm_tb,
    meta_filter,
    nn_evo,
    nn_h1,
    nn_profit,
    nn_seq,
    nn_slope,
    pattern_knn,
    shock_move,
    som_regime,
    squeeze_breakout,
    trend_regime,
)

__all__ = ["REGISTRY", "Strategy", "WalkForwardScores", "clip_score", "create", "register",
           "load_oos_scores", "save_oos_scores", "oos_path"]
