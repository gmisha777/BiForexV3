"""Intrabar order-flow features: what happened inside each trading-timeframe bar, from 1m data.

Every feature of bar t uses only the 1-minute bars inside [t, t + timeframe) and, for the
context columns, earlier bars, so it is known when bar t closes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schema import timeframe_delta

MIN_MINUTES_SHARE = 0.5  # bars with less 1m data than this share are left empty
CONTEXT_BARS = 42  # 7 days of 4h bars


def flow_features(m1: pd.DataFrame, timeframe: str = "4h") -> pd.DataFrame:
    """m1: stored 1m klines (open_time, open, high, low, close, volume, trades,
    taker_buy_volume). Returns float32 features indexed by the bar open_time."""
    bar = timeframe_delta(timeframe)
    per_bar = int(bar / pd.Timedelta("1min"))
    d = m1.sort_values("open_time").set_index("open_time")
    t = d.index
    key = t.floor(bar)
    minute = ((t - key) // pd.Timedelta("1min")).to_numpy()

    close, vol = d["close"].to_numpy(), d["volume"].to_numpy()
    buy = d["taker_buy_volume"].to_numpy()
    r = np.diff(np.log(close), prepend=np.nan)
    r = np.nan_to_num(r)
    imb = 2 * buy - vol
    last_hour = minute >= per_bar - per_bar // 4
    first_half, last_30 = minute < per_bar // 2, minute >= per_bar - per_bar // 8

    cols = {
        "n": np.ones(len(d)), "vol": vol, "buy": buy, "r": r, "r2": r**2, "r3": r**3,
        "r4": r**4, "absr": np.abs(r), "up": (r > 0).astype(float), "pv": close * vol,
        "trades": d["trades"].to_numpy(dtype=float),
        "vol_lh": vol * last_hour, "buy_lh": buy * last_hour,
        "trades_lh": d["trades"].to_numpy(dtype=float) * last_hour,
        "imb_h1": imb * first_half, "vol_h1": vol * first_half,
        "imb_h2": imb * ~first_half, "vol_h2": vol * ~first_half,
        "r_lh": r * last_hour, "r_l30": r * last_30,
    }
    agg = pd.DataFrame(cols, index=key).groupby(level=0).sum()
    ohlc = pd.DataFrame({"high": d["high"].to_numpy(), "low": d["low"].to_numpy(),
                         "close": close}, index=key).groupby(level=0).agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"))
    hi_pos = d["high"].groupby(key).idxmax()
    lo_pos = d["low"].groupby(key).idxmin()
    vol_sq = pd.Series((vol**2), index=key).groupby(level=0).sum()

    n = agg["n"]
    mean = agg["r"] / n
    var = (agg["r2"] / n - mean**2).clip(lower=1e-18)
    std = np.sqrt(var)
    m3 = agg["r3"] / n - 3 * mean * agg["r2"] / n + 2 * mean**3
    m4 = (agg["r4"] / n - 4 * mean * agg["r3"] / n + 6 * mean**2 * agg["r2"] / n
          - 3 * mean**4)
    rng = (ohlc["high"] - ohlc["low"]).replace(0, np.nan)
    rv = std * np.sqrt(n)  # realized volatility of the bar
    vwap = agg["pv"] / agg["vol"].replace(0, np.nan)

    f = pd.DataFrame(index=agg.index)
    f["taker_ratio"] = agg["buy"] / agg["vol"]
    f["taker_ratio_lh"] = agg["buy_lh"] / agg["vol_lh"]
    f["imb_shift"] = agg["imb_h2"] / agg["vol_h2"] - agg["imb_h1"] / agg["vol_h1"]
    f["vol_share_lh"] = agg["vol_lh"] / agg["vol"]
    f["trades_share_lh"] = agg["trades_lh"] / agg["trades"]
    f["vol_concentration"] = vol_sq / agg["vol"] ** 2 * n  # 1 = even, higher = bursts
    f["ret_bar_rv"] = agg["r"] / rv
    f["ret_lh_rv"] = agg["r_lh"] / rv
    f["ret_l30_rv"] = agg["r_l30"] / rv
    f["vwap_dev_range"] = (ohlc["close"] - vwap) / rng
    f["close_pos_range"] = (ohlc["close"] - ohlc["low"]) / rng
    f["skew"] = m3 / var**1.5
    f["kurt"] = m4 / var**2
    f["max_move_rv"] = pd.Series(np.abs(r), index=key).groupby(level=0).max() / rv
    f["up_share"] = agg["up"] / n
    f["t_high"] = ((hi_pos - hi_pos.index) / bar).astype(float)
    f["t_low"] = ((lo_pos - lo_pos.index) / bar).astype(float)
    f["rv_pct"] = rv
    f["range_rv"] = np.log(rng / ohlc["close"]) - np.log(rv)

    # context over previous bars (rolling windows include the current bar: it has closed)
    k = CONTEXT_BARS
    f["rv_rel"] = np.log(rv / rv.rolling(k, min_periods=k // 2).mean())
    f["vol_rel"] = np.log(agg["vol"] / agg["vol"].rolling(k, min_periods=k // 2).mean())
    tr = f["taker_ratio"]
    f["taker_ratio_z"] = (tr - tr.rolling(k, min_periods=k // 2).mean()) / tr.rolling(
        k, min_periods=k // 2).std()
    imb_bar = (2 * agg["buy"] - agg["vol"])
    f["imb_1d"] = imb_bar.rolling(6).sum() / agg["vol"].rolling(6).sum()
    f["imb_3d"] = imb_bar.rolling(18).sum() / agg["vol"].rolling(18).sum()

    f = f[n >= MIN_MINUTES_SHARE * per_bar]
    f = f.replace([np.inf, -np.inf], np.nan).astype("float32")
    f.index.name = "open_time"
    return f
