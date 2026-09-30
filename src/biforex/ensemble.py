"""Ensemble of strategies: weighted mean of scores and the entry/exit rules on it.

combined = sum(w_i * score_i) / sum(w_i) over the strategies that produced a score, so a
strategy that fails for a bar is left out and the remaining weights are renormalized.
The same functions drive the backtest and the live bot.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

from .config import Settings
from .strategies import Strategy, create, load_oos_scores

log = logging.getLogger(__name__)


def resolve_weights(settings: Settings, names: list[str] | None = None,
                    weights: list[float] | None = None) -> dict[str, float]:
    """Strategies and weights: from the arguments if given, else the enabled ones in config."""
    if names is None:
        chosen = settings.enabled_strategies
    else:
        if weights is None:
            weights = [settings.strategies[n].weight if n in settings.strategies else 1.0
                       for n in names]
        if len(weights) != len(names):
            raise ValueError("number of weights differs from number of strategies")
        chosen = dict(zip(names, weights, strict=True))
    chosen = {n: float(w) for n, w in chosen.items() if w > 0}
    if not chosen:
        raise ValueError("no strategy enabled (strategies: ... enabled: true, weight > 0)")
    return chosen


def combine(scores: dict[str, float | None], weights: dict[str, float]) -> float | None:
    """Weighted mean of the available scores; None if no strategy produced one."""
    num = den = 0.0
    for name, w in weights.items():
        s = scores.get(name)
        if s is None or (isinstance(s, float) and math.isnan(s)):
            continue
        num += w * s
        den += w
    return num / den if den > 0 else None


def combine_frames(frames: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """Offline version of `combine` for walk-forward scores.
    frames: strategy -> DataFrame(index open_time, columns symbol, score).
    Returns index open_time with columns symbol, score, n_strategies and one column per strategy.
    """
    wide = None
    for name in weights:
        f = frames[name].reset_index().rename(columns={"score": name})
        f = f[["open_time", "symbol", name]].drop_duplicates(["open_time", "symbol"])
        wide = f if wide is None else wide.merge(f, on=["open_time", "symbol"], how="outer")
    names = list(weights)
    w = pd.Series(weights)
    values = wide[names]
    mask = values.notna()
    num = (values.fillna(0) * w).sum(axis=1)
    den = (mask * w).sum(axis=1)
    wide["score"] = num / den.where(den > 0)
    wide["n_strategies"] = mask.sum(axis=1)
    wide = wide.dropna(subset=["score"]).sort_values(["open_time", "symbol"])
    return wide.set_index("open_time")


def entry_side(score: float, enter_score: float) -> int:
    """+1 long, -1 short, 0 nothing."""
    if score >= enter_score:
        return 1
    if score <= -enter_score:
        return -1
    return 0


def exit_signal(position_side: int, score: float, exit_score: float) -> bool:
    """True when the score points against the open position strongly enough."""
    return position_side * score <= -exit_score


def load_oos(settings: Settings, weights: dict[str, float]) -> pd.DataFrame:
    """Combined walk-forward scores of the chosen strategies (for backtests)."""
    return combine_frames({n: load_oos_scores(n) for n in weights}, weights)


class LiveEnsemble:
    """Enabled strategies with their saved models, scoring the latest closed bar."""

    def __init__(self, settings: Settings, weights: dict[str, float] | None = None):
        self.weights = weights or resolve_weights(settings)
        self.strategies: dict[str, Strategy] = {n: create(n, settings) for n in self.weights}

    def load(self) -> None:
        for s in self.strategies.values():
            s.load()
        log.info("ensemble: %s", ", ".join(f"{n} (weight {w:g})"
                                            for n, w in self.weights.items()))

    def score(self, symbol: str, bar: pd.Timestamp) -> tuple[float | None, dict]:
        parts: dict[str, float | None] = {}
        for name, strategy in self.strategies.items():
            try:
                parts[name] = strategy.score(symbol, bar)
            except Exception:
                log.exception("%s %s: strategy failed, left out of the ensemble", name, symbol)
                parts[name] = None
        missing = [n for n, v in parts.items() if v is None]
        if missing:
            log.warning("%s: no score from %s; weights renormalized", symbol, missing)
        return combine(parts, self.weights), parts
