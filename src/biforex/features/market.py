"""Futures market features: funding rate, open interest, long/short ratios."""

from __future__ import annotations

import numpy as np
import pandas as pd

METRICS_PUBLISH_DELAY = pd.Timedelta("5min")


def asof_join(
    decision_times: pd.Series,
    other: pd.DataFrame,
    time_col: str,
    tolerance: pd.Timedelta | None = None,
    allow_exact: bool = True,
) -> pd.DataFrame:
    """For each decision time take the latest row of `other` known at that moment."""
    left = pd.DataFrame({"_t": decision_times.to_numpy()})
    right = other.sort_values(time_col).rename(columns={time_col: "_t"})
    right["_t"] = right["_t"].astype(left["_t"].dtype)
    merged = pd.merge_asof(
        left, right, on="_t", direction="backward",
        tolerance=tolerance, allow_exact_matches=allow_exact,
    )
    return merged.drop(columns="_t").set_index(decision_times.index)


def funding_features(funding: pd.DataFrame, decision_times: pd.Series) -> pd.DataFrame:
    """funding: columns funding_time, funding_rate. The bot runs seconds after the bar closes,
    which may be before a funding settled at that same second is published, so only strictly
    earlier funding events are used."""
    fr = funding.sort_values("funding_time").reset_index(drop=True)
    events = pd.DataFrame({
        "funding_time": fr["funding_time"],
        "funding_last": fr["funding_rate"],
        "funding_mean3": fr["funding_rate"].rolling(3).mean(),
        "funding_sum21": fr["funding_rate"].rolling(21).sum(),  # ~7 days
    })
    return asof_join(
        decision_times, events, "funding_time", tolerance=pd.Timedelta("9h"), allow_exact=False)


def metrics_features(
    metrics: pd.DataFrame, decision_times: pd.Series, max_staleness: pd.Timedelta,
    quote_volume: pd.Series,
) -> pd.DataFrame:
    """metrics: 5-minute rows (time, sum_open_interest, ...) in the archive time convention,
    where a value is published about 5 minutes after its timestamp. A row counts as known only
    strictly after time + METRICS_PUBLISH_DELAY."""
    m = metrics.sort_values("time").set_index("time")
    taker_1h = m["sum_taker_long_short_vol_ratio"].rolling("1h").mean()
    rows = pd.DataFrame({
        "time": m.index + METRICS_PUBLISH_DELAY,
        "oi": m["sum_open_interest"].to_numpy(),
        "oi_value": m["sum_open_interest_value"].to_numpy(),
        "top_ls_accounts": m["count_toptrader_long_short_ratio"].to_numpy(),
        "top_ls_positions": m["sum_toptrader_long_short_ratio"].to_numpy(),
        "global_ls": m["count_long_short_ratio"].to_numpy(),
        "taker_ls_1h": taker_1h.to_numpy(),
    })
    j = asof_join(decision_times, rows, "time", tolerance=max_staleness, allow_exact=False)

    log_oi = np.log(j["oi"])
    f = pd.DataFrame(index=j.index)
    for k in (1, 4, 24):
        f[f"oi_chg_{k}"] = log_oi - log_oi.shift(k)
    f["oi_z168"] = (log_oi - log_oi.rolling(168, min_periods=120).mean()) / log_oi.rolling(
        168, min_periods=120).std()
    f["oi_to_volume24"] = np.log(j["oi_value"] / quote_volume.rolling(24).sum())
    for col in ("top_ls_accounts", "top_ls_positions", "global_ls", "taker_ls_1h"):
        lr = np.log(j[col])
        f[f"{col}_log"] = lr
        f[f"{col}_chg24"] = lr - lr.shift(24)
    return f
