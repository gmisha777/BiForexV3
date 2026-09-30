"""Out-of-sample evaluation of walk-forward predictions.

Signal statistics treat every bar as a separate trade (trades overlap), so they measure signal
quality, not account returns; the backtest (stage 5) simulates one position at a time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from . import CLASSES, PROBA_COLUMNS

THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60)


def round_trip_cost(fee_taker: float, slippage_bps: float) -> float:
    """Market entry + market exit."""
    return 2 * fee_taker + 2 * slippage_bps / 10_000


def probability_quality(preds: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Log-loss of the model vs. predicting the training class frequencies, per year and total."""
    y = preds["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    p_model = preds[list(PROBA_COLUMNS)].to_numpy()
    prior_cols = [f"prior_{c}" for c in PROBA_COLUMNS]
    p_prior = folds.loc[preds["fold"], prior_cols].to_numpy()

    def row(mask: np.ndarray) -> dict:
        ll_m = log_loss(y[mask], p_model[mask], labels=[0, 1, 2])
        ll_p = log_loss(y[mask], p_prior[mask], labels=[0, 1, 2])
        acc = (p_model[mask].argmax(1) == y[mask]).mean()
        return {"rows": int(mask.sum()), "logloss_model": ll_m, "logloss_prior": ll_p,
                "improvement_pct": 100 * (ll_p - ll_m) / ll_p, "accuracy": acc}

    years = preds.index.year
    out = {str(yr): row(np.asarray(years == yr)) for yr in sorted(set(years))}
    out["total"] = row(np.ones(len(preds), dtype=bool))
    return pd.DataFrame(out).T


def _trade_stats(ret: pd.Series, outcome: pd.Series, cost: float, total_bars: int) -> dict:
    if ret.empty:
        return {"signals": 0}
    net = ret - cost
    monthly = net.groupby(net.index.strftime("%Y-%m")).mean()
    return {
        "signals": len(ret),
        "share_of_bars_pct": 100 * len(ret) / total_bars,
        "tp_rate_pct": 100 * (outcome == 1).mean(),
        "gross_mean_pct": 100 * ret.mean(),
        "net_mean_pct": 100 * net.mean(),
        "months_positive_pct": 100 * (monthly > 0).mean(),
    }


def signal_table(
    preds: pd.DataFrame, cost: float, thresholds: tuple[float, ...] = THRESHOLDS
) -> pd.DataFrame:
    """Long when p_up >= th (and p_up > p_down); short mirrored. Plus 'every bar' baselines."""
    n = len(preds)
    rows = {
        ("long", "every bar"): _trade_stats(preds["long_ret"], preds["long_outcome"], cost, n),
        ("short", "every bar"): _trade_stats(preds["short_ret"], preds["short_outcome"], cost, n),
    }
    for th in thresholds:
        lg = preds[(preds["p_up"] >= th) & (preds["p_up"] > preds["p_down"])]
        sh = preds[(preds["p_down"] >= th) & (preds["p_down"] > preds["p_up"])]
        rows[("long", f"p>={th:.2f}")] = _trade_stats(lg["long_ret"], lg["long_outcome"], cost, n)
        rows[("short", f"p>={th:.2f}")] = _trade_stats(
            sh["short_ret"], sh["short_outcome"], cost, n)
    return pd.DataFrame(rows).T


def trend_baseline(preds: pd.DataFrame, X: pd.DataFrame, cost: float) -> pd.DataFrame:
    """Simple rule for comparison: long when EMA21 > EMA55, short otherwise."""
    x = X.loc[preds.index]
    up = x["ema55_dist_atr"] > x["ema21_dist_atr"]  # (c-ema55) > (c-ema21)  <=>  ema21 > ema55
    n = len(preds)
    return pd.DataFrame({
        ("long", "EMA21>EMA55"): _trade_stats(
            preds.loc[up, "long_ret"], preds.loc[up, "long_outcome"], cost, n),
        ("short", "EMA21<EMA55"): _trade_stats(
            preds.loc[~up, "short_ret"], preds.loc[~up, "short_outcome"], cost, n),
    }).T


def calibration(preds: pd.DataFrame, column: str = "p_up", bins: int = 8) -> pd.DataFrame:
    """Predicted probability vs. realized frequency of the class."""
    cls = {"p_up": 1, "p_down": -1, "p_flat": 0}[column]
    edges = np.unique(np.quantile(preds[column], np.linspace(0, 1, bins + 1)))
    bucket = pd.cut(preds[column], edges, include_lowest=True)
    hit = preds["label"] == cls
    return pd.DataFrame({
        "rows": bucket.value_counts(sort=False),
        "predicted": preds[column].groupby(bucket, observed=False).mean(),
        "realized": hit.groupby(bucket, observed=False).mean(),
    })


def top_features(importance: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    share = importance.div(importance.sum(axis=1), axis=0)
    return pd.DataFrame({
        "gain_share_pct": 100 * share.mean(),
        "folds_in_top20_pct": 100 * (share.rank(axis=1, ascending=False) <= 20).mean(),
    }).sort_values("gain_share_pct", ascending=False).head(n)
