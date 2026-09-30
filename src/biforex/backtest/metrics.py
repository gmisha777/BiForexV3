"""Performance statistics for a backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import BacktestResult


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def summary(result: BacktestResult, initial_equity: float) -> dict:
    eq, tr = result.equity, result.trades
    days = (eq.index[-1] - eq.index[0]).days or 1
    final = eq.iloc[-1]
    daily = eq.resample("1D").last().ffill().pct_change().dropna()
    downside = daily[daily < 0].std()
    out = {
        "start": eq.index[0].date(), "end": eq.index[-1].date(),
        "final_equity": final,
        "total_return_pct": 100 * (final / initial_equity - 1),
        "cagr_pct": 100 * ((final / initial_equity) ** (365 / days) - 1),
        "max_drawdown_pct": 100 * max_drawdown(eq),
        "sharpe": np.sqrt(365) * daily.mean() / daily.std() if daily.std() > 0 else np.nan,
        "sortino": np.sqrt(365) * daily.mean() / downside if downside > 0 else np.nan,
        "trades": len(tr),
        "halted_at": result.halted_at,
    }
    if len(tr):
        wins, losses = tr.loc[tr["pnl"] > 0, "pnl"], tr.loc[tr["pnl"] <= 0, "pnl"]
        out.update({
            "win_rate_pct": 100 * len(wins) / len(tr),
            "profit_factor": wins.sum() / -losses.sum() if len(losses) else np.inf,
            "avg_trade_pct_of_equity": 100 * tr["pnl"].mean() / initial_equity,
            "avg_bars_held": tr["bars"].mean(),
            "fees_total": tr["fees"].sum(),
            "funding_total": tr["funding"].sum(),
            "trades_per_month": len(tr) / (days / 30.4),
        })
    return out


def breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """PnL by symbol and side."""
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby(["symbol", "side"])
    return pd.DataFrame({
        "trades": g.size(),
        "win_rate_pct": 100 * g["pnl"].apply(lambda x: (x > 0).mean()),
        "pnl": g["pnl"].sum(),
        "fees": g["fees"].sum(),
        "funding": g["funding"].sum(),
    })


def exit_reasons(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby("reason")
    return pd.DataFrame({"trades": g.size(), "pnl": g["pnl"].sum(), "avg_pnl": g["pnl"].mean()})


def yearly(equity: pd.Series) -> pd.DataFrame:
    by_year = equity.groupby(equity.index.year)
    first, last = by_year.first(), by_year.last()
    start = first.copy()
    start.iloc[1:] = last.iloc[:-1].to_numpy()  # each year starts at the previous year's end
    return pd.DataFrame({
        "return_pct": 100 * (last / start - 1),
        "max_drawdown_pct": by_year.apply(lambda e: 100 * max_drawdown(e)),
    })


def buy_and_hold(bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    c = bars.set_index("open_time")["close"]
    c = c[(c.index >= start) & (c.index <= end)]
    return {"return_pct": 100 * (c.iloc[-1] / c.iloc[0] - 1),
            "max_drawdown_pct": 100 * max_drawdown(c)}
