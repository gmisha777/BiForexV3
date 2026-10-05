"""Causal technical indicators on pandas Series (no look-ahead: each value uses past rows only)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing used by RSI and ATR."""
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift()
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = wilder(delta.clip(lower=0), n)
    down = wilder(-delta.clip(upper=0), n)
    return 100 - 100 / (1 + up / down)


def stochastic_k(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14, smooth: int = 3
) -> pd.Series:
    lowest, highest = low.rolling(n).min(), high.rolling(n).max()
    return (100 * (close - lowest) / (highest - lowest)).rolling(smooth).mean()


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    lowest, highest = low.rolling(n).min(), high.rolling(n).max()
    return -100 * (highest - close) / (highest - lowest)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    mean = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - mean) / (0.015 * mad)


def mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 14
) -> pd.Series:
    tp = (high + low + close) / 3
    flow = tp * volume
    pos = flow.where(tp > tp.shift(), 0.0).rolling(n).sum()
    neg = flow.where(tp < tp.shift(), 0.0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg)


def parkinson_vol(high: pd.Series, low: pd.Series, n: int) -> pd.Series:
    hl = np.log(high / low) ** 2
    return np.sqrt(hl.rolling(n).mean() / (4 * np.log(2)))


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Average Directional Index: trend strength 0..100 regardless of direction."""
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_ = wilder(true_range(high, low, close), n)
    plus_di = 100 * wilder(plus_dm, n) / atr_
    minus_di = 100 * wilder(minus_dm, n) / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return wilder(dx, n)
