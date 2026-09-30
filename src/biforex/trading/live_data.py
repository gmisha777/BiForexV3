"""Bring local data up to the last closed bar and score it with the strategy ensemble."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
import requests

from ..config import Settings
from ..data import rest, store
from ..data.update import update_funding, update_klines, update_metrics
from ..ensemble import LiveEnsemble
from ..features import indicators as ind

log = logging.getLogger(__name__)


def refresh_data(settings: Settings, symbols: list[str], session: requests.Session) -> None:
    """Klines (trading timeframe) + funding + metrics; metrics tail via REST."""
    tf = settings.timeframe
    for s in symbols:
        update_klines(settings, s, tf, session)
        update_funding(settings, s, session)
        update_metrics(settings, s, session)
        path = store.metrics_path(settings.data.dir, s)
        start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=settings.bot.metrics_rest_hours)
        tail = rest.fetch_metrics(s, int(start.timestamp() * 1000), session)
        if not tail.empty:
            store.save(store.merge([store.load(path), tail], "time"), path)


@dataclass
class Decision:
    """Ensemble view of one closed bar for one symbol."""

    bar: pd.Timestamp  # open time of the closed bar
    score: float  # combined score, -1..+1
    close: float
    atr: float  # for stop / take-profit distances (same ATR as the labels)
    parts: dict[str, float | None] = field(default_factory=dict)  # per-strategy scores


class Scorer:
    """Scores the latest closed bar with the enabled strategies."""

    def __init__(self, settings: Settings, ensemble: LiveEnsemble | None = None):
        self.settings = settings
        self.ensemble = ensemble or LiveEnsemble(settings)
        self.ensemble.load()

    def market(self, symbol: str, bar: pd.Timestamp) -> tuple[float, float] | None:
        """Close and ATR of the bar."""
        s = self.settings
        k = store.load(store.klines_path(s.data.dir, symbol, s.timeframe))
        k = k.set_index("open_time")
        if bar not in k.index:
            return None
        atr = ind.atr(k["high"], k["low"], k["close"], s.label.atr_period)
        return float(k.at[bar, "close"]), float(atr.at[bar])

    def decide(self, symbol: str, bar: pd.Timestamp) -> Decision | None:
        market = self.market(symbol, bar)
        if market is None:
            log.warning("%s: bar %s not in local data", symbol, bar)
            return None
        score, parts = self.ensemble.score(symbol, bar)
        if score is None:
            log.warning("%s: no strategy produced a score for %s", symbol, bar)
            return None
        close, atr = market
        log.info("%s bar %s close %.2f ATR %.2f | ensemble score %+.3f %s", symbol, bar, close,
                 atr, score, {n: None if v is None else round(v, 3) for n, v in parts.items()})
        return Decision(bar, score, close, atr, parts)
