import pandas as pd
import pytest

from biforex.backtest import metrics
from biforex.backtest.engine import BacktestParams, run_backtest

T0 = pd.Timestamp("2024-01-01", tz="UTC")
BAR = pd.Timedelta("4h")


def make_bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    t = pd.date_range(T0, periods=len(rows), freq="4h")
    o, h, lo, c = zip(*rows, strict=True)
    return pd.DataFrame({"open_time": t, "open": o, "high": h, "low": lo, "close": c})


def preds_for(n: int, signals: dict[int, str], atr: float = 2.0, symbol="X") -> pd.DataFrame:
    """signals: bar index -> 'up' / 'down'."""
    t = pd.date_range(T0, periods=n, freq="4h")
    p = pd.DataFrame({"symbol": symbol, "score": 0.0, "atr": atr}, index=t)
    for i, s in signals.items():
        p.iloc[i, p.columns.get_loc("score")] = 0.3 if s == "up" else -0.3
    return p


FLAT = (100.0, 100.5, 99.5, 100.0)
NO_COSTS = dict(fee=0.0, slippage_bps=0.0, risk_pct=1.0, max_leverage=100, horizon=5,
                initial_equity=10_000, bar=BAR)


def run(rows, signals, **kw):
    params = BacktestParams(**{**NO_COSTS, **kw})
    bars = {"X": make_bars(rows)}
    return run_backtest(bars, preds_for(len(rows), signals), params)


def test_long_take_profit_earns_two_r():
    # entry at close 100 (bar 0), ATR 2 -> SL 98, TP 104; bar 2 reaches 104
    r = run([FLAT, FLAT, (100, 104.5, 99.8, 104), FLAT, FLAT, FLAT], {0: "up"})
    t = r.trades.iloc[0]
    assert (t["side"], t["reason"]) == ("long", "take")
    assert t["qty"] == pytest.approx(10_000 * 0.01 / 2)  # risk 100 USDT / 2 USDT stop
    assert t["pnl"] == pytest.approx(200)  # 2R
    assert r.equity.iloc[-1] == pytest.approx(10_200)


def test_short_stop_loses_one_r_with_costs():
    r = run([FLAT, (100, 102.5, 99.9, 102), FLAT, FLAT], {0: "down"},
            fee=0.0005, slippage_bps=2)
    t = r.trades.iloc[0]
    assert (t["side"], t["reason"]) == ("short", "stop")
    assert t["pnl"] < -100  # 1R + fees + slippage
    assert t["fees"] == pytest.approx(t["qty"] * (t["entry"] + t["exit"]) * 0.0005)


def test_time_exit_after_horizon():
    r = run([FLAT] * 10, {0: "up"}, horizon=3)
    t = r.trades.iloc[0]
    assert (t["reason"], t["bars"]) == ("time", 3)
    assert t["exit_time"] == T0 + 4 * BAR  # close of bar 3


def test_opposite_signal_reverses():
    r = run([FLAT] * 6, {0: "up", 2: "down"})
    assert list(r.trades["reason"][:1]) == ["signal"]
    assert list(r.trades["side"][:2]) == ["long", "short"]


def test_leverage_cap_limits_size():
    r = run([FLAT] * 4, {0: "up"}, max_leverage=1, risk_pct=5)
    assert r.trades.iloc[0]["notional"] == pytest.approx(10_000, rel=1e-6)


def test_funding_charged_to_long():
    rows = [FLAT] * 4
    bars = {"X": make_bars(rows)}
    funding = {"X": pd.DataFrame({"funding_time": [T0 + BAR + pd.Timedelta("1h")],
                                  "funding_rate": [0.001]})}
    params = BacktestParams(**{**NO_COSTS, "horizon": 2})
    r = run_backtest(bars, preds_for(4, {0: "up"}), params, funding)
    t = r.trades.iloc[0]
    assert t["funding"] == pytest.approx(t["qty"] * 100 * 0.001)
    assert t["pnl"] == pytest.approx(-t["funding"])


def test_max_drawdown_halts_new_entries():
    # every trade is stopped out; with 10% risk the 15% drawdown limit trips after two losses
    loss_bar = (100, 100.2, 97, 100)
    rows = [FLAT] + [loss_bar, FLAT] * 5
    signals = {i: "up" for i in range(0, 11, 2)}
    r = run(rows, signals, risk_pct=10, max_dd_pct=15, daily_loss_pct=100)
    assert r.halted_at is not None
    assert len(r.trades) == 2


LIMIT = dict(entry="limit", limit_offset_atr=0.25, maker_fee=0.0002, fee=0.0005,
             take_profit_limit=True)


def test_limit_entry_fills_and_take_profit_is_maker():
    # limit 99.5 (close 100 - 0.25 * ATR 2); bar 1 dips to 99.4 -> filled at 99.5;
    # SL 97.5, TP 103.5 reached in bar 2
    r = run([FLAT, (100, 100.2, 99.4, 100), (100, 103.8, 99.8, 103.5), FLAT, FLAT],
            {0: "up"}, **LIMIT)
    t = r.trades.iloc[0]
    assert (t["reason"], t["entry"], t["exit"]) == ("take", 99.5, 103.5)
    assert t["fees"] == pytest.approx(t["qty"] * (99.5 + 103.5) * 0.0002)
    assert (r.limit_orders, r.limit_fills) == (1, 1)


def test_limit_entry_expires_when_price_runs_away():
    r = run([FLAT, (100.2, 103, 99.8, 102.5), FLAT, FLAT], {0: "up"}, **LIMIT)
    assert r.trades.empty
    assert (r.limit_orders, r.limit_fills) == (1, 0)


def test_limit_fill_then_stop_in_same_bar_via_minutes():
    rows = [FLAT, (100, 100.2, 97, 97.2), FLAT, FLAT]
    fine_t = pd.date_range(T0 + BAR, periods=240, freq="1min")
    fine = pd.DataFrame({"open_time": fine_t, "open": 100.0, "high": 100.1, "low": 99.9,
                         "close": 100.0})
    fine.loc[10, "low"] = 99.4  # limit 99.5 filled in minute 10
    fine.loc[50, "low"] = 97.0  # stop 97.5 in minute 50
    params = BacktestParams(**{**NO_COSTS, **LIMIT})
    r = run_backtest({"X": make_bars(rows)}, preds_for(4, {0: "up"}), params,
                     intrabar={"X": fine})
    assert list(r.trades["reason"]) == ["stop"]


def test_metrics_summary_runs():
    r = run([FLAT, FLAT, (100, 104.5, 99.8, 104), FLAT, FLAT, FLAT], {0: "up"})
    s = metrics.summary(r, 10_000)
    assert s["trades"] == 1 and s["win_rate_pct"] == 100
    assert s["total_return_pct"] == pytest.approx(2.0)


def test_stop_limit_fills_at_stop_without_slippage():
    rows = [FLAT, (100, 100.5, 97, 98), FLAT, FLAT]  # stop 98 touched inside bar 1
    base = dict(fee=0.0005, slippage_bps=10)
    market = run(rows, {0: "up"}, **base).trades.iloc[0]
    stop_limit = run(rows, {0: "up"}, **base, stop_order="stop_limit").trades.iloc[0]
    stop = stop_limit["entry"] - 2.0  # 1 ATR below the (slipped) entry
    assert stop_limit["exit"] == pytest.approx(stop) and market["exit"] < stop


def test_stop_limit_gap_beyond_offset_closes_at_market_open():
    rows = [FLAT, (95, 95.5, 94, 95), FLAT]  # opens far below the stop 98 (limit 97.8)
    t = run(rows, {0: "up"}, stop_order="stop_limit", slippage_bps=10).trades.iloc[0]
    assert t["reason"] == "stop" and t["exit"] < 95  # market fill at the open, with slippage


def test_limit_exit_is_maker_when_next_bar_trades_through():
    rows = [FLAT] * 3 + [(100, 100.6, 99.6, 100.2), FLAT]
    r = run(rows, {0: "up"}, horizon=2, exit_limit=True, fee=0.0005, maker_fee=0.0002)
    t = r.trades.iloc[0]
    assert t["reason"] == "time" and t["exit"] == pytest.approx(100.0)  # close of bar 2
    assert t["fees"] == pytest.approx(t["qty"] * 100 * 0.0002 + t["qty"] * t["entry"] * 0.0005)


def test_limit_exit_falls_back_to_market_at_next_close():
    rows = [FLAT] * 3 + [(99.9, 99.95, 99.0, 99.2), FLAT]  # never trades back up to 100
    t = run(rows, {0: "up"}, horizon=2, exit_limit=True, slippage_bps=0).trades.iloc[0]
    assert t["exit"] == pytest.approx(99.2)


def test_portfolio_limits_cap_simultaneous_positions():
    t = pd.date_range(T0, periods=6, freq="4h")
    preds = pd.concat([
        pd.DataFrame({"symbol": sym, "score": [0.3] + [0.0] * 5, "atr": 2.0}, index=t)
        for sym in ("A", "B", "C")])
    bars = {sym: make_bars([FLAT] * 6) for sym in ("A", "B", "C")}
    params = BacktestParams(**{**NO_COSTS, "horizon": 5})
    assert len(run_backtest(bars, preds, params).trades) == 3
    params.max_same_side = 2
    assert len(run_backtest(bars, preds, params).trades) == 2
    params.max_same_side, params.max_positions = 0, 1
    assert len(run_backtest(bars, preds, params).trades) == 1
