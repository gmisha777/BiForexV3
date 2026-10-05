"""Strategy flow_1m: LightGBM on intrabar order flow computed from 1-minute bars.

lgbm_tb sees each bar only as open/high/low/close/volume; this strategy looks inside the bar:
buyer/seller imbalance and how it shifts, the last hour's move and volume, distance of the
close from VWAP, realized volatility, skew and tails of 1m returns, when the high and low
happened. Only symbols with stored 1m data are used (BTCUSDT, ETHUSDT).
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from ..config import PROJECT_ROOT
from ..data import store
from ..features.flow import flow_features
from . import register
from .lgbm_base import LgbmStrategy

log = logging.getLogger(__name__)

LIVE_DAYS = 60  # 1m history used live: enough for the 7-day context windows


@register
class Flow1m(LgbmStrategy):
    name = "flow_1m"
    data_intervals = ("1m",)

    def _m1_path(self, symbol: str):
        return store.klines_path(self.st.data.dir, symbol, "1m")

    def _cache_path(self, symbol: str):
        return (PROJECT_ROOT / "data" / "strategies" / self.name / self.st.timeframe
                / f"{symbol}.parquet")

    def _has_data(self, symbol: str) -> bool:
        return self._m1_path(symbol).exists()

    def features(self, symbol: str, live: bool = False) -> pd.DataFrame:
        m1_path = self._m1_path(symbol)
        if live:
            m1 = pd.read_parquet(m1_path)
            m1 = m1[m1["open_time"] >= m1["open_time"].iloc[-1] - pd.Timedelta(days=LIVE_DAYS)]
            return flow_features(m1, self.st.timeframe)
        cache = self._cache_path(symbol)
        if cache.exists() and os.path.getmtime(cache) >= os.path.getmtime(m1_path):
            return pd.read_parquet(cache)
        log.info("%s %s: computing intrabar features from 1m data...", self.name, symbol)
        f = flow_features(pd.read_parquet(m1_path), self.st.timeframe)
        cache.parent.mkdir(parents=True, exist_ok=True)
        f.to_parquet(cache)
        return f
