"""Strategy pattern_knn: trade a 4h pattern by what followed similar patterns on other scales.

Idea: price patterns are self-similar across timeframes. At each 4h bar close the last `n`
4h bars form a pattern; among bars of a library scale (1m, 5m, 15m, 1h built from the 1m
data of all symbols, or 4h itself as a control) the `k` most similar n-bar windows are found
and what followed each of them is measured with the shared exit rules on that scale (take
profit tp_atr x ATR, stop sl_atr x ATR, time limit horizon_bars bars; ATR(14) of that scale;
a bar touching both counts as the stop). The neighbors' mean of (long R - short R), minus the
library's mean (the drift), is the raw signal.

Patterns are scale-free: closes relative to the last close, wicks and bodies, all divided by
the window's mean bar range (relative), plus volume relative to the window mean ("full";
"path" uses the closes only). Euclidean distance.

Library: the traded symbols ("own") or also data.extra_symbols ("all").

No look-ahead: a library window is used only if its outcome was complete by the close of the
4h bar being scored. Parameters (scale, n, k, pattern) are chosen on in-sample data only, the
score is raw / q x 0.10 with q fitted in-sample (see rule_base).
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..data import store
from ..data.schema import timeframe_delta
from ..strategies.base import calibration_scale
from . import register
from .rule_base import RuleStrategy

log = logging.getLogger(__name__)

ATR_BARS = 14
MIN_LIBRARY = 20  # x k: fewer usable library windows -> no signal
QUERY_CHUNK = pd.Timedelta("7D")  # queries in a chunk share the library cutoff (conservative)
LIB_BLOCK = 200_000
SCALES = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}


@dataclass
class Bars:
    """OHLCV arrays of one symbol on one scale (gaps are not filled)."""

    time: np.ndarray  # open time, int64 ns
    o: np.ndarray
    h: np.ndarray
    lo: np.ndarray
    c: np.ndarray
    v: np.ndarray
    step: int  # ns per bar

    @classmethod
    def from_frame(cls, df: pd.DataFrame, step: pd.Timedelta) -> Bars:
        df = df.dropna(subset=["close"])
        return cls(pd.DatetimeIndex(df["open_time"]).as_unit("ns").asi8,
                   *(df[c].to_numpy(dtype=np.float64)
                     for c in ("open", "high", "low", "close", "volume")), step.value)


def resample(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = m1.set_index("open_time").sort_index()
    if rule == "1min":
        return d.reset_index()
    g = d.groupby(d.index.floor(rule)).agg(open=("open", "first"), high=("high", "max"),
                                           low=("low", "min"), close=("close", "last"),
                                           volume=("volume", "sum"), n=("close", "size"))
    full = int(pd.Timedelta(rule) / pd.Timedelta("1min"))
    g = g[g["n"] >= 0.8 * full]  # mostly complete bars only
    return g.drop(columns="n").rename_axis("open_time").reset_index()


def patterns(b: Bars, ends: np.ndarray, n: int, kind: str) -> np.ndarray:
    """Scale-free pattern vectors of the n-bar windows ending at bar index `ends`."""
    idx = ends[:, None] + np.arange(1 - n, 1)[None, :]
    lc, lh, ll, lo = (np.log(x[idx]) for x in (b.c, b.h, b.lo, b.o))
    rng = np.maximum((lh - ll).mean(axis=1, keepdims=True), 1e-6)
    parts = [(lc[:, :-1] - lc[:, -1:]) / rng]
    if kind == "full":
        lv = np.log1p(b.v[idx])
        parts += [0.5 * (lh - np.maximum(lc, lo)) / rng, 0.5 * (np.minimum(lc, lo) - ll) / rng,
                  0.5 * (lc - lo) / rng, 0.3 * (lv - lv.mean(axis=1, keepdims=True))]
    return np.clip(np.concatenate(parts, axis=1), -20, 20).astype(np.float32)


def contiguous(b: Bars, ends: np.ndarray, n: int) -> np.ndarray:
    return b.time[ends] - b.time[ends - (n - 1)] == (n - 1) * b.step


def outcomes(b: Bars, ends: np.ndarray, tp: float, sl: float, horizon: int) -> np.ndarray:
    """long R - short R of trades entered at the close of bar `ends` (exits as label)."""
    tr = np.maximum(b.h[1:], b.c[:-1]) - np.minimum(b.lo[1:], b.c[:-1])
    atr = pd.Series(np.concatenate([[np.nan], tr])).rolling(ATR_BARS).mean().to_numpy()
    c0, a = b.c[ends], atr[ends]
    a = np.where(a > 1e-5 * c0, a, np.nan)  # flat bars (no trading): no outcome
    res = {}
    with np.errstate(invalid="ignore"):
        for side in (1, -1):
            res[side] = _side_r(b, ends, c0, a, side, tp, sl, horizon)
    return res[1] - res[-1]


def _side_r(b: Bars, ends, c0, a, side: int, tp: float, sl: float, horizon: int) -> np.ndarray:
    r = np.full(len(ends), np.nan)
    open_ = np.ones(len(ends), bool)
    for j in range(1, horizon + 1):
        hi, low = b.h[ends + j], b.lo[ends + j]
        fav = (hi - c0) / a if side == 1 else (c0 - low) / a
        adv = (c0 - low) / a if side == 1 else (hi - c0) / a
        stop, take = open_ & (adv >= sl), open_ & (fav >= tp)
        r[stop] = -sl
        r[take & ~stop] = tp
        open_ &= ~(stop | take)
    r[open_] = side * (b.c[ends + horizon][open_] - c0[open_]) / a[open_]
    return r


class Library:
    """Pattern windows of all symbols on one scale with outcomes, sorted by availability."""

    def __init__(self, bars: list[Bars], n: int, kind: str, stride: int, tp: float, sl: float,
                 horizon: int):
        X, Y, T = [], [], []
        for b in bars:
            ends = np.arange(max(n, ATR_BARS + 1), len(b.c) - horizon, stride)
            ends = ends[contiguous(b, ends, n)
                        & (b.time[ends + horizon] - b.time[ends] == horizon * b.step)]
            y = outcomes(b, ends, tp, sl, horizon)
            ok = np.isfinite(y)
            ends = ends[ok]
            X.append(patterns(b, ends, n, kind)), Y.append(y[ok])
            T.append(b.time[ends + horizon] + b.step)  # outcome known at this time
        order = np.argsort(np.concatenate(T), kind="stable")
        self.X = torch.from_numpy(np.concatenate(X)[order])
        self.Y = np.concatenate(Y)[order].astype(np.float32)
        self.T = np.concatenate(T)[order]
        self.sq = (self.X * self.X).sum(1)
        self.cum_y = np.concatenate([[0.0], np.cumsum(self.Y, dtype=np.float64)])

    def query(self, Q: np.ndarray, times: np.ndarray, ks: list[int]) -> dict[int, np.ndarray]:
        """Per k: mean outcome of the k nearest available windows minus the available mean."""
        k = max(ks)
        out = {kk: np.full(len(Q), np.nan) for kk in ks}
        order = np.argsort(times)
        q_all = torch.from_numpy(Q)
        chunk = QUERY_CHUNK.value
        start = 0
        while start < len(order):
            t0 = times[order[start]]
            stop = np.searchsorted(times[order], t0 + chunk, side="left")
            sel = order[start:stop]
            start = stop
            avail = int(np.searchsorted(self.T, t0, side="right"))
            if avail < MIN_LIBRARY * k:
                continue
            q = q_all[sel]
            qsq = (q * q).sum(1, keepdim=True)
            best_d, best_i = None, None
            for a in range(0, avail, LIB_BLOCK):
                b = min(avail, a + LIB_BLOCK)
                d = qsq - 2 * q @ self.X[a:b].T + self.sq[a:b][None, :]
                kk = min(k, b - a)
                dv, di = torch.topk(d, kk, dim=1, largest=False)
                di = di + a
                if best_d is not None:
                    dv, di = torch.cat([best_d, dv], 1), torch.cat([best_i, di], 1)
                    dv, pos = torch.topk(dv, min(k, dv.shape[1]), dim=1, largest=False)
                    di = torch.gather(di, 1, pos)
                best_d, best_i = dv, di
            drift = self.cum_y[avail] / avail
            y = self.Y[best_i.numpy()]  # neighbors sorted nearest first
            for kk in ks:
                out[kk][sel] = y[:, :kk].mean(axis=1) - drift
        return out


@register
class PatternKnn(RuleStrategy):
    name = "pattern_knn"
    data_intervals = ("1m",)
    defaults = {"scale": ["1m", "5m", "15m", "1h", "4h"], "n": [8, 16, 32], "k": [50, 200],
                "pattern": ["path", "full"], "library": ["own"], "top_share": [0.05],
                "min_in_sample_trades": 30}

    def __init__(self, settings, params: dict | None = None):
        super().__init__(settings, params)
        self._bars: dict[tuple[str, str], Bars] = {}
        self._raw: dict[tuple, pd.DataFrame] = {}

    def grid(self) -> list[dict]:
        keys = ("scale", "n", "k", "pattern", "library", "top_share")
        return [dict(zip(keys, v, strict=True))
                for v in itertools.product(*(self.cfg[k] for k in keys))]

    # --- data ----------------------------------------------------------------------------
    def _library_symbols(self, params: dict) -> list[str]:
        """"own": the traded symbols; "all": also data.extra_symbols. A library on the
        trading timeframe uses its klines, smaller scales are built from 1m data."""
        st, d = self.settings, self.settings.data.dir
        symbols = st.symbols if params.get("library", "own") == "own" else st.all_symbols
        interval = st.timeframe if params["scale"] == st.timeframe else "1m"
        return [s for s in symbols if store.klines_path(d, s, interval).exists()]

    def _get_bars(self, symbol: str, scale: str, live: bool = False) -> Bars:
        key = (symbol, scale)
        if key not in self._bars or live:
            d, tf = self.settings.data.dir, self.settings.timeframe
            if scale == tf:
                df = store.load(store.klines_path(d, symbol, tf))
            else:
                df = resample(store.load(store.klines_path(d, symbol, "1m")), SCALES[scale])
            self._bars[key] = Bars.from_frame(df, pd.Timedelta(SCALES[scale]))
        return self._bars[key]

    def _compute(self, params: dict, live: bool = False) -> pd.DataFrame:
        """Raw signal of every traded symbol for one (scale, n, k, pattern)."""
        lab = self.settings.label
        n, scale = params["n"], params["scale"]
        stride = 1 if scale in ("1h", "4h") else max(1, n // 4)
        lib = Library([self._get_bars(s, scale, live) for s in self._library_symbols(params)], n,
                      params["pattern"], stride, lab.tp_atr, lab.sl_atr, lab.horizon_bars)
        tf = timeframe_delta(self.settings.timeframe)
        parts = []
        for s in self.settings.symbols:
            b = self._get_bars(s, self.settings.timeframe, live)
            ends = np.arange(n - 1, len(b.c))
            if live:
                ends = ends[-6:]
            ends = ends[contiguous(b, ends, n)]
            raw = lib.query(patterns(b, ends, n, params["pattern"]),
                            b.time[ends] + tf.value, list(self.cfg["k"]))
            idx = pd.to_datetime(b.time[ends], utc=True)
            parts.append(pd.DataFrame({"symbol": s, **{f"k{k}": v for k, v in raw.items()}},
                                      index=idx))
        log.info("%s %s: library %d windows", self.name, params, len(lib.Y))
        return pd.concat(parts)

    # --- rule interface --------------------------------------------------------------------
    def raw(self, symbol: str, params: dict, live: bool = False) -> pd.Series:
        key = (params["scale"], params["n"], params["pattern"], params.get("library", "own"))
        if live:
            r = self._compute(params, live=True)
        else:
            if key not in self._raw:
                self._raw[key] = self._compute(params)
            r = self._raw[key]
        r = r[r["symbol"] == symbol][f"k{params['k']}"]
        r.index.name = "open_time"
        return r

    def fit(self, raw_in_sample: pd.Series, params: dict) -> dict:
        share = params.get("top_share", self.cfg["top_share"][0])
        return {"q": calibration_scale(raw_in_sample, share)}

    def to_score(self, raw: pd.Series, params: dict, fitted: dict) -> pd.Series:
        return raw / fitted["q"] * 0.10
