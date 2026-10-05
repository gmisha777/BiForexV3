"""Backtest the ensemble on out-of-sample walk-forward scores (run scripts/train.py first).

Usage:
    python scripts/backtest.py                                  # strategies/weights from config
    python scripts/backtest.py --strategies lgbm_tb other --weights 1 2
    python scripts/backtest.py --equity 20 --exchange-limits    # small account, Binance minimums
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
import requests

from biforex.backtest import metrics
from biforex.backtest.engine import params_from, run_backtest
from biforex.config import PROJECT_ROOT, Settings, load_settings
from biforex.data import store
from biforex.ensemble import load_oos, resolve_weights
from biforex.labels.build import labels_path
from biforex.log import setup_logging
from biforex.trading.rules import SymbolRules

log = logging.getLogger("backtest")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def attach_atr(settings: Settings, preds: pd.DataFrame) -> pd.DataFrame:
    """ATR at each bar (for stop / take profit distances) from the shared labels."""
    parts = []
    for symbol, g in preds.groupby("symbol"):
        lab = store.load(labels_path(settings, symbol)).set_index("open_time")["atr"]
        parts.append(g.join(lab, how="inner"))
    return pd.concat(parts).dropna(subset=["atr"]).sort_index(kind="stable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("--equity", type=float, help="starting equity (default from config)")
    parser.add_argument("--exchange-limits", action="store_true",
                        help="apply Binance minimum order sizes (as the live bot does)")
    parser.add_argument("--strategies", nargs="+", help="strategies to combine (default: config)")
    parser.add_argument("--weights", nargs="+", type=float, help="weights for --strategies")
    parser.add_argument("--any-coverage", action="store_true",
                        help="also use bars scored by only some of the strategies")
    args = parser.parse_args()
    settings = load_settings(args.config)
    if args.equity:
        settings.backtest.initial_equity = args.equity
    setup_logging(settings.logging.level, settings.logging.dir)

    weights = resolve_weights(settings, args.strategies, args.weights)
    preds = load_oos(settings, weights)
    preds = preds[preds["symbol"].isin(settings.symbols)]  # extra symbols are not traded
    if not args.any_coverage:  # compare strategies on the period they all cover
        preds = preds[preds["n_strategies"] == len(weights)]
    symbols = sorted(preds["symbol"].unique())
    preds = attach_atr(settings, preds)
    name = "+".join(f"{n}x{w:g}" for n, w in weights.items())
    report_dir = PROJECT_ROOT / "reports" / "ensemble" / settings.report_key / name
    log.info("ensemble %s: %d scored bars %s -> %s; bars scored by all strategies: %.0f%%",
             weights, len(preds), preds.index[0].date(), preds.index[-1].date(),
             100 * (preds["n_strategies"] == len(weights)).mean())
    d, tf = settings.data.dir, settings.timeframe
    bars = {s: store.load(store.klines_path(d, s, tf)) for s in symbols}
    funding = {s: store.load(store.funding_path(d, s)) for s in symbols}
    intrabar = {s: store.load(store.klines_path(d, s, settings.label.intrabar_interval))
                for s in symbols}
    cols = ["symbol", "score", "atr"]

    rules = None
    if args.exchange_limits:
        info = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=30).json()
        rules = {s: SymbolRules.from_exchange_info(info, s) for s in symbols}
        log.info("exchange limits: %s", {s: (r.min_qty, r.min_notional) for s, r in
                                         rules.items()})

    sc = settings.ensemble
    main_params = params_from(settings, sc.enter_score, sc.exit_score, rules)
    result = run_backtest(bars, preds[cols], main_params, funding, intrabar)
    out = report_dir / "backtest"
    out.mkdir(parents=True, exist_ok=True)
    result.trades.to_csv(out / "trades.csv", index=False)
    result.equity.to_csv(out / "equity.csv")

    s = metrics.summary(result, main_params.initial_equity)
    log.info("backtest %s -> %s | enter |score|>=%.2f, exit %.2f | risk %.1f%%/trade | "
             "entry %s", s["start"], s["end"], sc.enter_score, sc.exit_score,
             main_params.risk_pct, main_params.entry)
    log.info("summary:\n%s", pd.Series(s).to_string())
    if result.limit_orders:
        log.info("limit entries: %d placed, %d filled (%.0f%%)", result.limit_orders,
                 result.limit_fills, 100 * result.limit_fills / result.limit_orders)
    log.info("by year:\n%s", metrics.yearly(result.equity).round(2).to_string())
    log.info("by symbol and side:\n%s", metrics.breakdown(result.trades).round(2).to_string())
    log.info("exit reasons:\n%s", metrics.exit_reasons(result.trades).round(2).to_string())
    for sym in symbols:
        bh = metrics.buy_and_hold(bars[sym], result.equity.index[0], result.equity.index[-1])
        log.info("buy & hold %s: return %.1f%%, max drawdown %.1f%%", sym,
                 bh["return_pct"], bh["max_drawdown_pct"])

    grid = {}
    for th in settings.backtest.threshold_grid:
        r = run_backtest(bars, preds[cols], params_from(settings, th, th, rules), funding,
                         intrabar)
        m = metrics.summary(r, main_params.initial_equity)
        grid[th] = {k: m.get(k) for k in ("trades", "total_return_pct", "cagr_pct",
                                           "max_drawdown_pct", "sharpe", "win_rate_pct",
                                           "profit_factor", "halted_at")}
    grid_df = pd.DataFrame(grid).T
    grid_df.index.name = "threshold"
    grid_df.to_csv(out / "threshold_grid.csv")
    log.info("threshold sensitivity (enter = exit = threshold):\n%s", grid_df.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
