"""Build labels for a symbol from stored data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Settings
from ..data import store
from ..data.schema import timeframe_delta
from .triple_barrier import triple_barrier


def build_labels(settings: Settings, symbol: str) -> pd.DataFrame:
    lc, tf = settings.label, settings.timeframe
    klines = store.load(store.klines_path(settings.data.dir, symbol, tf))
    if klines is None:
        raise FileNotFoundError(f"no {tf} klines for {symbol}; run scripts/download_history.py")
    intrabar = store.load(store.klines_path(settings.data.dir, symbol, lc.intrabar_interval))
    return triple_barrier(
        klines, horizon=lc.horizon_bars, tp_atr=lc.tp_atr, sl_atr=lc.sl_atr,
        atr_period=lc.atr_period, intrabar=intrabar, bar_length=timeframe_delta(tf),
    )


def labels_path(settings: Settings, symbol: str) -> Path:
    return settings.data.dir / "labels" / settings.timeframe / f"{symbol}.parquet"
