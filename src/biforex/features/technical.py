"""Price and volume features from OHLCV bars. All features are scale-free (ratios, ATR units)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schema import timeframe_delta
from . import indicators as ind

RETURN_LAGS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]  # Fibonacci, as in the legacy MT4 expert
EMA_SPANS = [8, 21, 55, 144, 377, 610]
SLOPE_SPANS = [21, 55, 144]


def price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature set for the trading timeframe. df: OHLCV indexed by open_time."""
    o, h, lo, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    logc = np.log(c)
    atr14 = ind.atr(h, lo, c, 14)
    f: dict[str, pd.Series] = {}

    for k in RETURN_LAGS:
        f[f"ret_{k}"] = logc - logc.shift(k)

    for n in EMA_SPANS:
        f[f"ema{n}_dist_atr"] = (c - ind.ema(c, n)) / atr14
    for n in SLOPE_SPANS:
        e = ind.ema(c, n)
        f[f"ema{n}_slope_atr"] = (e - e.shift(5)) / atr14
    macd = ind.ema(c, 12) - ind.ema(c, 26)
    signal = ind.ema(macd, 9)
    f["macd_atr"] = macd / atr14
    f["macd_hist_atr"] = (macd - signal) / atr14

    f["rsi14"] = ind.rsi(c, 14)
    f["rsi24"] = ind.rsi(c, 24)
    f["stoch_k"] = ind.stochastic_k(h, lo, c, 14, 3)
    f["williams_r"] = ind.williams_r(h, lo, c, 14)
    f["cci20"] = ind.cci(h, lo, c, 20)
    f["mfi14"] = ind.mfi(h, lo, c, v, 14)

    r1 = logc.diff()
    f["atr_pct"] = atr14 / c
    f["rv24"] = r1.rolling(24).std()
    f["rv168"] = r1.rolling(168).std()
    f["rv_ratio"] = f["rv24"] / f["rv168"]
    f["parkinson24"] = ind.parkinson_vol(h, lo, 24)
    sma20, std20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_width"] = 4 * std20 / sma20
    f["bb_pos"] = (c - sma20) / (2 * std20)

    f["vol_rel24"] = np.log(v / v.rolling(24).mean())
    f["vol_rel168"] = np.log(v / v.rolling(168).mean())
    taker = df["taker_buy_volume"] / v
    f["taker_buy_ratio"] = taker
    f["taker_imbalance24"] = ((2 * taker - 1) * v).rolling(24).sum() / v.rolling(24).sum()
    f["trades_rel24"] = np.log(df["trades"] / df["trades"].rolling(24).mean())

    rng = h - lo
    f["body_ratio"] = (c - o) / rng
    f["upper_wick"] = (h - np.maximum(o, c)) / rng
    f["lower_wick"] = (np.minimum(o, c) - lo) / rng
    f["range_atr"] = rng / atr14

    for n in (24, 168):
        hh, ll = h.rolling(n).max(), lo.rolling(n).min()
        f[f"dist_high{n}_atr"] = (hh - c) / atr14
        f[f"dist_low{n}_atr"] = (c - ll) / atr14
        f[f"range_pos{n}"] = (c - ll) / (hh - ll)

    return pd.DataFrame(f, index=df.index)


def compact_features(df: pd.DataFrame) -> pd.DataFrame:
    """Small feature set for higher timeframes."""
    h, lo, c = df["high"], df["low"], df["close"]
    logc = np.log(c)
    atr14 = ind.atr(h, lo, c, 14)
    return pd.DataFrame({
        "ret_1": logc.diff(),
        "ret_3": logc - logc.shift(3),
        "rsi14": ind.rsi(c, 14),
        "ema21_dist_atr": (c - ind.ema(c, 21)) / atr14,
        "atr_pct": atr14 / c,
    }, index=df.index)


def context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features of the market leader (BTC) added to other symbols."""
    logc = np.log(df["close"])
    r1 = logc.diff()
    return pd.DataFrame({
        "ret_1": r1,
        "ret_4": logc - logc.shift(4),
        "ret_24": logc - logc.shift(24),
        "rv24": r1.rolling(24).std(),
    }, index=df.index)


def time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour + index.minute / 60
    dow = index.dayofweek
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
    }, index=index)


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate bars to a higher timeframe; bars are labelled by their open time (UTC grid)."""
    agg = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "taker_buy_volume": "sum", "trades": "sum",
    }
    rule = timeframe_delta(timeframe)
    out = df.resample(rule, origin="epoch", label="left", closed="left").agg(agg)
    return out.dropna(subset=["close"])
