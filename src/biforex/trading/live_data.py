"""Bring local data up to the last closed bar and predict it with the current model."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from ..config import Settings
from ..data import rest, store
from ..data.update import update_funding, update_klines, update_metrics
from ..features.pipeline import build_features
from ..model import PROBA_COLUMNS
from ..model.registry import POOLED, load_current

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
class Prediction:
    bar: pd.Timestamp  # open time of the closed bar
    p_down: float
    p_flat: float
    p_up: float
    atr: float
    close: float

    @property
    def probs(self) -> tuple[float, float, float]:
        return self.p_down, self.p_flat, self.p_up


class Predictor:
    def __init__(self, settings: Settings):
        self.settings = settings
        name = POOLED if settings.model.pooled else settings.symbols[0]
        self.booster, self.meta = load_current(name)
        if self.meta["timeframe"] != settings.timeframe:
            raise RuntimeError(f"model timeframe {self.meta['timeframe']} != config "
                               f"{settings.timeframe}; retrain")
        log.info("model %s/%s: trained until %s, %d trees, symbols %s", name,
                 self.meta["version"], self.meta["trained_until"], self.meta["trees"],
                 self.meta["symbols"])

    def predict(self, symbol: str, bar: pd.Timestamp) -> Prediction | None:
        """Prediction for the bar with open time `bar`, or None if its data is not there."""
        features = build_features(self.settings, symbol)
        if bar not in features.index:
            log.warning("%s: no features for bar %s (last %s)", symbol, bar, features.index[-1])
            return None
        row = features.loc[[bar]].copy()
        if "symbol_id" in self.meta["features"]:
            row["symbol_id"] = np.float32(self.meta["symbols"].index(symbol))
        X = row.reindex(columns=self.meta["features"])
        p = self.booster.predict(X)[0]
        klines = store.load(store.klines_path(self.settings.data.dir, symbol,
                                              self.settings.timeframe)).set_index("open_time")
        close = float(klines.at[bar, "close"])
        atr = float(row["atr_pct"].iloc[0]) * close
        pred = Prediction(bar, *map(float, p), atr=atr, close=close)
        log.info("%s bar %s close %.2f ATR %.2f | %s", symbol, bar, close, atr,
                 " ".join(f"{c}={v:.3f}" for c, v in zip(PROBA_COLUMNS, p, strict=True)))
        return pred
