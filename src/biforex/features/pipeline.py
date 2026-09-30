"""Assemble the feature table for one symbol.

Row t is indexed by the bar's open_time; the decision is taken at open_time + timeframe, when the
bar has closed. Every joined source (higher timeframes, funding, metrics) is aligned to that
decision time, so nothing from the future leaks in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Settings
from ..data import store
from ..data.schema import timeframe_delta
from . import market, technical


def compute_features(
    klines: pd.DataFrame,
    timeframe: str = "1h",
    funding: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
    context: pd.DataFrame | None = None,
    higher_timeframes: tuple[str, ...] = ("4h", "1d"),
    metrics_max_staleness: str = "3h",
) -> pd.DataFrame:
    """klines/context: frames with open_time + OHLCV columns (as stored). Returns float32 features
    indexed by open_time. Optional sources passed as None are skipped; periods a source does not
    cover (e.g. open interest before 2021) stay NaN, which LightGBM handles natively."""
    with np.errstate(divide="ignore", invalid="ignore"):  # log(0), x/0 -> inf -> NaN below
        return _compute(klines, timeframe, funding, metrics, context, higher_timeframes,
                        metrics_max_staleness)


def _compute(klines, timeframe, funding, metrics, context, higher_timeframes,
             metrics_max_staleness) -> pd.DataFrame:
    bars = klines.set_index("open_time").sort_index()
    decision = pd.Series(bars.index + timeframe_delta(timeframe), index=bars.index)
    parts = [technical.price_features(bars), technical.time_features(bars.index)]

    for tf in higher_timeframes:
        htf_bars = technical.resample_ohlcv(bars, tf)
        htf = technical.compact_features(htf_bars).add_prefix(f"h{tf}_")
        htf["available_at"] = htf.index + timeframe_delta(tf)  # known once the HTF bar has closed
        parts.append(market.asof_join(decision, htf.reset_index(drop=True), "available_at"))

    if funding is not None and not funding.empty:
        parts.append(market.funding_features(funding, decision))

    if metrics is not None and not metrics.empty:
        parts.append(market.metrics_features(
            metrics, decision, pd.Timedelta(metrics_max_staleness), bars["quote_volume"]))

    if context is not None and not context.empty:
        ctx_bars = context.set_index("open_time").sort_index()
        ctx = technical.context_features(ctx_bars).reindex(bars.index)
        own_r1 = np.log(bars["close"]).diff()
        own_r24 = np.log(bars["close"]) - np.log(bars["close"]).shift(24)
        ctx["rel_ret_24"] = own_r24 - ctx["ret_24"]
        ctx["corr168"] = own_r1.rolling(168).corr(ctx["ret_1"])
        parts.append(ctx.add_prefix("ctx_"))

    features = pd.concat(parts, axis=1)
    features = features.replace([np.inf, -np.inf], np.nan).astype("float32")
    features.index.name = "open_time"
    return features


def build_features(settings: Settings, symbol: str) -> pd.DataFrame:
    """Load stored data for a symbol and compute its feature table (warm-up rows dropped)."""
    data_dir, tf, fc = settings.data.dir, settings.timeframe, settings.features
    klines = store.load(store.klines_path(data_dir, symbol, tf))
    if klines is None:
        raise FileNotFoundError(f"no {tf} klines for {symbol}; run scripts/download_history.py")
    context = None
    if fc.context_symbol and fc.context_symbol != symbol:
        context = store.load(store.klines_path(data_dir, fc.context_symbol, tf))

    features = compute_features(
        klines,
        timeframe=tf,
        funding=store.load(store.funding_path(data_dir, symbol)),
        metrics=store.load(store.metrics_path(data_dir, symbol)),
        context=context,
        higher_timeframes=tuple(fc.higher_timeframes),
        metrics_max_staleness=fc.metrics_max_staleness,
    )
    return features.iloc[fc.warmup_bars:]


def features_path(settings: Settings, symbol: str) -> Path:
    return settings.data.dir / "features" / settings.timeframe / f"{symbol}.parquet"
