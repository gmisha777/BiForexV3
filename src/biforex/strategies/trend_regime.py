"""Strategy trend_regime: follow the trend, but only when the market is trending (no ML).

raw = (EMA_fast - EMA_slow) / ATR               trend direction and strength
      set to 0 when ADX < adx_min               no clear trend -> no view
      or when the close is on the other side    price not confirming the trend
         of EMA_fast
score = raw / q * 0.10, q calibrated so that |score| >= 0.10 on the `top_share` of
in-sample bars with the strongest trend. Parameters and q: in-sample only (see rule_base).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ..data import store
from ..features import indicators as ind
from . import register
from .base import calibration_scale
from .rule_base import RuleStrategy


def raw_signal(k: pd.DataFrame, fast: int, slow: int, adx_min: float) -> pd.Series:
    """k: klines indexed by open_time. Trend strength in ATR units, 0 outside trends."""
    h, lo, c = k["high"], k["low"], k["close"]
    atr = ind.atr(h, lo, c, 14)
    ema_fast, ema_slow = ind.ema(c, fast), ind.ema(c, slow)
    gap = (ema_fast - ema_slow) / atr
    trending = ind.adx(h, lo, c, 14) >= adx_min
    confirmed = np.sign(c - ema_fast) == np.sign(gap)
    return gap.where(trending & confirmed, 0.0).where(gap.notna())


@register
class TrendRegime(RuleStrategy):
    name = "trend_regime"
    defaults = {"top_share": 0.06,
                "fast_slow": [[21, 89], [50, 200]], "adx_min": [20, 25]}

    def grid(self) -> list[dict]:
        return [{"fast": f, "slow": s, "adx_min": a} for (f, s), a in
                itertools.product(self.cfg["fast_slow"], self.cfg["adx_min"])]

    def raw(self, symbol: str, params: dict, live: bool = False) -> pd.Series:
        k = store.load(store.klines_path(self.settings.data.dir, symbol,
                                         self.settings.timeframe)).set_index("open_time")
        return raw_signal(k, params["fast"], params["slow"], params["adx_min"])

    def fit(self, raw_in_sample: pd.Series, params: dict) -> dict:
        return {"q": calibration_scale(raw_in_sample, self.cfg["top_share"])}

    def to_score(self, raw: pd.Series, params: dict, fitted: dict) -> pd.Series:
        return raw / fitted["q"] * 0.10
