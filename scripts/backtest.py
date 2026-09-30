"""Backtest the out-of-sample walk-forward predictions (run scripts/train.py first).

Usage:
    python scripts/backtest.py
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd
import requests

from biforex.backtest import metrics
from biforex.backtest.engine import BacktestParams, run_backtest
from biforex.config import PROJECT_ROOT, Settings, load_settings
from biforex.data import store
from biforex.data.schema import timeframe_delta
from biforex.log import setup_logging
from biforex.model.registry import POOLED
from biforex.trading.rules import SymbolRules

log = logging.getLogger("backtest")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def params_from(settings: Settings, enter: float, exit_: float,
                rules: dict[str, SymbolRules] | None = None) -> BacktestParams:
    return BacktestParams(
        initial_equity=settings.backtest.initial_equity,
        risk_pct=settings.risk.per_trade_pct,
        max_leverage=settings.risk.max_leverage,
        fee=settings.execution.fee_taker,
        maker_fee=settings.execution.fee_maker,
        slippage_bps=settings.execution.slippage_bps,
        entry=settings.execution.entry_order,
        limit_offset_atr=settings.execution.limit_offset_atr,
        take_profit_limit=settings.execution.take_profit_limit,
        tp_atr=settings.label.tp_atr,
        sl_atr=settings.label.sl_atr,
        horizon=settings.label.horizon_bars,
        enter_prob=enter,
        exit_prob=exit_,
        daily_loss_pct=settings.risk.daily_loss_pct,
        max_dd_pct=settings.risk.max_dd_pct,
        bar=timeframe_delta(settings.timeframe),
        rules=rules,
        max_trade_risk_pct=settings.risk.max_trade_risk_pct,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("--equity", type=float, help="starting equity (default from config)")
    parser.add_argument("--exchange-limits", action="store_true",
                        help="apply Binance minimum order sizes (as the live bot does)")
    args = parser.parse_args()
    settings = load_settings(args.config)
    if args.equity:
        settings.backtest.initial_equity = args.equity
    setup_logging(settings.logging.level, settings.logging.dir)

    name = POOLED if settings.model.pooled else settings.symbols[0]
    report_dir = PROJECT_ROOT / "reports" / name
    preds = pd.read_parquet(report_dir / "oos_predictions.parquet")
    preds = preds[preds["symbol"].isin(settings.symbols)]  # extra symbols are not traded
    symbols = sorted(preds["symbol"].unique())
    d, tf = settings.data.dir, settings.timeframe
    bars = {s: store.load(store.klines_path(d, s, tf)) for s in symbols}
    funding = {s: store.load(store.funding_path(d, s)) for s in symbols}
    intrabar = {s: store.load(store.klines_path(d, s, settings.label.intrabar_interval))
                for s in symbols}
    cols = ["symbol", "p_up", "p_down", "atr"]

    rules = None
    if args.exchange_limits:
        info = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=30).json()
        rules = {s: SymbolRules.from_exchange_info(info, s) for s in symbols}
        log.info("exchange limits: %s", {s: (r.min_qty, r.min_notional) for s, r in
                                         rules.items()})

    sc = settings.signal
    main_params = params_from(settings, sc.enter_prob, sc.exit_prob, rules)
    result = run_backtest(bars, preds[cols], main_params, funding, intrabar)
    out = report_dir / "backtest"
    out.mkdir(parents=True, exist_ok=True)
    result.trades.to_csv(out / "trades.csv", index=False)
    result.equity.to_csv(out / "equity.csv")

    s = metrics.summary(result, main_params.initial_equity)
    log.info("backtest %s -> %s | enter p>=%.2f, exit p>=%.2f | risk %.1f%%/trade | "
             "entry %s", s["start"], s["end"], sc.enter_prob, sc.exit_prob,
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
