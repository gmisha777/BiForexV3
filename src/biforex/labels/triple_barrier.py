"""Triple-barrier labels.

For every bar t a position is opened at close[t] on both sides with
    take profit = tp_atr * ATR[t],  stop loss = sl_atr * ATR[t],  time limit = horizon bars,
and we record which barrier is hit first. With tp_atr > sl_atr the long and short take profits
are mutually exclusive, which gives the 3-class target:
    label = 1 (long TP first), -1 (short TP first), 0 (neither).

When one bar touches both TP and SL, the order is resolved on intrabar (1m) data; if that is
unavailable or both happen in the same minute, the stop is assumed first (pessimistic).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..features import indicators as ind

TP, SL, TIMEOUT = 1, -1, 0
ONE_HOUR = pd.Timedelta("1h")
LIMIT_THROUGH = 1e-4  # price must trade 1 bp beyond a limit order for it to count as filled


def limit_through(limit: float, is_long: bool) -> float:
    return limit * (1 - LIMIT_THROUGH) if is_long else limit * (1 + LIMIT_THROUGH)


def _to_ns(times: pd.Series) -> np.ndarray:
    """UTC timestamps -> int64 nanoseconds since epoch."""
    naive = times.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
    return naive.to_numpy().astype("int64")


class IntrabarResolver:
    """Tells whether TP or SL was touched first inside one bar, using finer bars."""

    def __init__(self, fine: pd.DataFrame | None, bar_length: pd.Timedelta):
        self.bar_ns = bar_length.value
        if fine is None or fine.empty:
            self.t = np.empty(0, dtype="int64")
            self.high = self.low = np.empty(0)
        else:
            self.t = _to_ns(fine["open_time"])
            self.high = fine["high"].to_numpy()
            self.low = fine["low"].to_numpy()

    def tp_first(self, bar_open_ns: int, tp: float, sl: float, is_long: bool) -> bool:
        i, j = np.searchsorted(self.t, [bar_open_ns, bar_open_ns + self.bar_ns])
        if i == j:
            return False
        hi, lo = self.high[i:j], self.low[i:j]
        tp_hit = hi >= tp if is_long else lo <= tp
        sl_hit = lo <= sl if is_long else hi >= sl
        first_tp = np.argmax(tp_hit) if tp_hit.any() else np.inf
        first_sl = np.argmax(sl_hit) if sl_hit.any() else np.inf
        return first_tp < first_sl  # same minute -> stop first

    def has_data(self, bar_open_ns: int) -> bool:
        i, j = np.searchsorted(self.t, [bar_open_ns, bar_open_ns + self.bar_ns])
        return j > i

    def limit_fill(
        self, bar_open_ns: int, limit: float, sl: float, tp: float, is_long: bool
    ) -> tuple[bool, str | None]:
        """Limit entry inside one bar: (filled, 'stop' | 'take' | None for the rest of the bar).
        A fill needs the price to trade through the limit by LIMIT_THROUGH (a touch may leave
        the order unfilled in the queue). The stop may trigger in the fill minute itself; the
        take only from the next minute."""
        i, j = np.searchsorted(self.t, [bar_open_ns, bar_open_ns + self.bar_ns])
        hi, lo = self.high[i:j], self.low[i:j]
        fill = lo <= limit_through(limit, is_long) if is_long else hi >= limit_through(
            limit, is_long)
        if not fill.any():
            return False, None
        k = int(np.argmax(fill))
        sl_hit = lo[k:] <= sl if is_long else hi[k:] >= sl
        tp_hit = hi[k + 1:] >= tp if is_long else lo[k + 1:] <= tp
        first_sl = np.argmax(sl_hit) if sl_hit.any() else np.inf
        first_tp = np.argmax(tp_hit) + 1 if tp_hit.any() else np.inf
        if first_sl == np.inf and first_tp == np.inf:
            return True, None
        return True, ("stop" if first_sl <= first_tp else "take")


def _first_hit(hits: np.ndarray) -> np.ndarray:
    """Index of the first True per row, or -1."""
    any_hit = hits.any(axis=1)
    return np.where(any_hit, hits.argmax(axis=1), -1)


def _side(
    entry: np.ndarray, dist_tp: np.ndarray, dist_sl: np.ndarray,
    fut_high: np.ndarray, fut_low: np.ndarray, fut_close_last: np.ndarray,
    bar_open_ns: np.ndarray, resolver: IntrabarResolver, is_long: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outcome, return and holding bars for one side. Rows: bars with a full horizon."""
    sign = 1 if is_long else -1
    tp = entry + sign * dist_tp
    sl = entry - sign * dist_sl
    if is_long:
        tp_bar = _first_hit(fut_high >= tp[:, None])
        sl_bar = _first_hit(fut_low <= sl[:, None])
    else:
        tp_bar = _first_hit(fut_low <= tp[:, None])
        sl_bar = _first_hit(fut_high >= sl[:, None])

    horizon = fut_high.shape[1]
    tp_k = np.where(tp_bar < 0, horizon, tp_bar)
    sl_k = np.where(sl_bar < 0, horizon, sl_bar)
    outcome = np.where(tp_k < sl_k, TP, np.where(sl_k < tp_k, SL, TIMEOUT))

    for r in np.flatnonzero((tp_k == sl_k) & (tp_k < horizon)):  # both in one bar
        bar_ns = bar_open_ns[r] + (tp_k[r] + 1) * resolver.bar_ns
        outcome[r] = TP if resolver.tp_first(bar_ns, tp[r], sl[r], is_long) else SL

    ret = np.select(
        [outcome == TP, outcome == SL],
        [dist_tp / entry, -dist_sl / entry],
        sign * (fut_close_last / entry - 1),
    )
    bars = np.select([outcome == TP, outcome == SL], [tp_k + 1, sl_k + 1], horizon)
    return outcome, ret, bars


def triple_barrier(
    klines: pd.DataFrame,
    horizon: int = 24,
    tp_atr: float = 2.0,
    sl_atr: float = 1.0,
    atr_period: int = 14,
    intrabar: pd.DataFrame | None = None,
    bar_length: pd.Timedelta = ONE_HOUR,
) -> pd.DataFrame:
    """klines: stored frame (open_time, open, high, low, close, ...). Returns labels indexed by
    open_time; the last `horizon` rows (future unknown) and ATR warm-up rows are NaN."""
    if tp_atr <= sl_atr:
        raise ValueError("tp_atr must exceed sl_atr so that long/short outcomes are exclusive")
    k = klines.sort_values("open_time").reset_index(drop=True)
    high, low, close = k["high"].to_numpy(), k["low"].to_numpy(), k["close"].to_numpy()
    atr = ind.atr(k["high"], k["low"], k["close"], atr_period).to_numpy()
    open_ns = _to_ns(k["open_time"])

    n = len(k) - horizon  # rows with a complete future window
    out = pd.DataFrame(index=pd.Index(k["open_time"], name="open_time"))
    out["atr"] = atr
    cols = ["label", "long_outcome", "long_ret", "long_bars",
            "short_outcome", "short_ret", "short_bars"]
    for c in cols:
        out[c] = np.nan
    if n <= 0:
        return out

    # the future window must be contiguous: a trading halt inside it would stretch the horizon
    contiguous = open_ns[horizon:] - open_ns[:n] == horizon * bar_length.value
    valid = ~np.isnan(atr[:n]) & contiguous
    rows = np.flatnonzero(valid)
    fut_high = sliding_window_view(high[1:], horizon)[:n][rows]
    fut_low = sliding_window_view(low[1:], horizon)[:n][rows]
    entry = close[rows]
    last_close = close[rows + horizon]
    resolver = IntrabarResolver(intrabar, bar_length)
    args = (entry, tp_atr * atr[rows], sl_atr * atr[rows], fut_high, fut_low, last_close,
            open_ns[rows], resolver)

    lo, lr, lb = _side(*args, is_long=True)
    so, sr, sb = _side(*args, is_long=False)
    label = np.where(lo == TP, 1, np.where(so == TP, -1, 0))

    idx = out.index[rows]
    out.loc[idx, cols] = np.column_stack([label, lo, lr, lb, so, sr, sb])
    return out
