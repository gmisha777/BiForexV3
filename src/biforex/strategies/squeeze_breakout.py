"""Strategy squeeze_breakout: trade the breakout of a range after volatility compression.

Squeeze: Bollinger width (20 bars, 2 sigma) was within its lowest `squeeze_pct` of the last
30 days at some point during the previous 6 bars. Breakout: the close leaves the range of the
previous `lookback` bars (long above the high, short below the low), optionally with volume
above its 24-bar mean. score = +-0.10 x (1 + distance beyond the range in ATR), 0 otherwise
(abstains). Parameters: in-sample only (see rule_base).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ..data import store
from ..features import indicators as ind
from . import register
from .rule_base import RuleStrategy

WINDOW = 180  # 30 days of 4h bars for the width percentile
RECENT = 6  # squeeze must have happened within the last day


def squeeze_signal(k: pd.DataFrame, squeeze_pct: float, lookback: int,
                   volume_confirm: bool) -> pd.Series:
    h, lo, c, v = k["high"], k["low"], k["close"], k["volume"]
    atr = ind.atr(h, lo, c, 14)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    width = 4 * sd / mid
    threshold = width.rolling(WINDOW, min_periods=WINDOW // 2).quantile(squeeze_pct)
    squeezed = (width <= threshold).astype(float).shift(1).rolling(RECENT).max() > 0
    top, bottom = h.shift(1).rolling(lookback).max(), lo.shift(1).rolling(lookback).min()
    up, down = (c - top) / atr, (bottom - c) / atr
    ok_volume = (v > v.rolling(24).mean()) if volume_confirm else True
    raw = pd.Series(0.0, index=k.index)
    long_ev = squeezed & (up > 0) & ok_volume
    short_ev = squeezed & (down > 0) & ok_volume
    raw[long_ev] = 1 + up[long_ev]
    raw[short_ev] = -(1 + down[short_ev])
    return raw.where(atr.notna() & threshold.notna())


@register
class SqueezeBreakout(RuleStrategy):
    name = "squeeze_breakout"
    defaults = {"squeeze_pct": [0.10, 0.20], "lookback": [20, 40],
                "volume_confirm": [True, False], "min_in_sample_trades": 20}

    def grid(self) -> list[dict]:
        return [{"squeeze_pct": p, "lookback": n, "volume_confirm": vc}
                for p, n, vc in itertools.product(self.cfg["squeeze_pct"],
                                                  self.cfg["lookback"],
                                                  self.cfg["volume_confirm"])]

    def raw(self, symbol: str, params: dict, live: bool = False) -> pd.Series:
        k = store.load(store.klines_path(self.settings.data.dir, symbol,
                                         self.settings.timeframe)).set_index("open_time")
        return squeeze_signal(k, params["squeeze_pct"], params["lookback"],
                              params["volume_confirm"])

    def to_score(self, raw: pd.Series, params: dict, fitted: dict) -> pd.Series:
        return np.sign(raw) * 0.10 * raw.abs().clip(upper=10)
