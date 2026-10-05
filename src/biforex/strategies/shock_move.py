"""Strategy shock_move: trade after a sharp move with a flush, fading OR following it (no ML).

Shock down (shock up mirrored):
  - the price fell >= move_atr ATR within 1-3 bars,
  - optionally RSI(14) <= 30 (>= 70 after a rise),
  - a flush: open interest dropped >= 2 % over 4 bars (forced closing) or volume >= 2x its
    24-bar mean (flush = "oi" or "volume").
direction = "fade" bets on a rebound (the original "exhaustion" idea), "follow" on
continuation. Fading was clearly negative already on in-sample data (PF 0.25-0.55), so both
are in the grid and the in-sample backtest picks one.
score = +-0.10 x move / move_atr on events (every event reaches the entry level), 0 otherwise:
the strategy abstains on normal bars. Parameters: in-sample only (see rule_base).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import register
from .lgbm_tb import LgbmTripleBarrier
from .rule_base import RuleStrategy

OI_DROP = -0.02  # log change of open interest over 4 bars
VOLUME_SPIKE = np.log(2.0)  # vol_rel24 is log(volume / 24-bar mean)


def shock_signal(f: pd.DataFrame, move_atr: float, flush: str, rsi_filter: bool) -> pd.Series:
    """f: lgbm_tb features. + after a dump, - after a pump (i.e. the "fade" direction)."""
    moves = pd.concat([f[f"ret_{k}"] / f["atr_pct"] for k in (1, 2, 3)], axis=1)
    drop, pump = -moves.min(axis=1), moves.max(axis=1)
    if flush == "oi":
        flushed = f["oi_chg_4"] <= OI_DROP
    else:
        flushed = f["vol_rel24"] >= VOLUME_SPIKE
    oversold = (f["rsi14"] <= 30) if rsi_filter else True
    overbought = (f["rsi14"] >= 70) if rsi_filter else True
    long_ev = (drop >= move_atr) & oversold & flushed
    short_ev = (pump >= move_atr) & overbought & flushed
    raw = pd.Series(0.0, index=f.index)
    raw[long_ev] = drop[long_ev] / move_atr
    raw[short_ev & ~long_ev] = -pump[short_ev & ~long_ev] / move_atr
    return raw.where(f["atr_pct"].notna())


@register
class ShockMove(RuleStrategy):
    name = "shock_move"
    defaults = {"move_atr": [2.0, 2.5, 3.0], "flush": ["oi", "volume"],
                "rsi_filter": [True, False], "direction": ["fade", "follow"],
                "min_in_sample_trades": 20}

    def __init__(self, settings, params: dict | None = None):
        super().__init__(settings, params)
        self.features_src = LgbmTripleBarrier(settings)

    def grid(self) -> list[dict]:
        return [{"move_atr": m, "flush": fl, "rsi_filter": r, "direction": d}
                for m, fl, r, d in itertools.product(self.cfg["move_atr"], self.cfg["flush"],
                                                     self.cfg["rsi_filter"],
                                                     self.cfg["direction"])]

    def raw(self, symbol: str, params: dict, live: bool = False) -> pd.Series:
        f = self.features_src.features(symbol, live)
        return shock_signal(f, params["move_atr"], params["flush"], params["rsi_filter"])

    def to_score(self, raw: pd.Series, params: dict, fitted: dict) -> pd.Series:
        sign = 1 if params["direction"] == "fade" else -1
        return sign * raw * 0.10
