"""Bar-by-bar account simulation.

Timeline for a bar with open_time t (it covers [t, t + bar)):
  1. a pending limit entry (placed at the previous close) is filled or expires;
     open positions: SL/TP orders resting on the exchange are checked against the bar's
     high/low (both touched -> intrabar data decides, else the stop is assumed first);
     funding settled inside the bar is charged;
  2. at the bar close the model prediction for row t becomes available:
     time exit (after `horizon` bars), exit when the ensemble score turns against the position
     (|score| >= exit_score), then new entries (|score| >= enter_score).
Entries are market orders (taker fee + slippage) or limit orders valid for one bar (maker fee,
filled only if the price trades through the limit). Stops, time and signal exits are market
orders; the take profit is a market order or, optionally, a resting limit (maker).
The account equity is shared by all symbols; each symbol holds at most one position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..ensemble import entry_side, exit_signal
from ..labels.triple_barrier import IntrabarResolver, _to_ns, limit_through
from ..trading.rules import SymbolRules, position_size


@dataclass
class BacktestParams:
    initial_equity: float = 5000.0
    risk_pct: float = 1.0  # of equity, lost if the stop is hit (before fees)
    max_leverage: float = 3.0
    fee: float = 0.0005  # taker, per side
    maker_fee: float = 0.0002
    slippage_bps: float = 2.0  # per market order
    entry: str = "market"  # "market" | "limit"
    limit_offset_atr: float = 0.0  # limit price = close -/+ offset * ATR (better than close)
    take_profit_limit: bool = False  # TP as resting limit order: maker fee, no slippage
    tp_atr: float = 2.0
    sl_atr: float = 1.0
    horizon: int = 12  # bars
    enter_score: float = 0.10  # |combined score| that opens a position
    exit_score: float = 0.10  # combined score against the position that closes it
    daily_loss_pct: float = 3.0  # no new entries for the rest of the UTC day
    max_dd_pct: float = 15.0  # stop opening positions for good
    bar: pd.Timedelta = field(default_factory=lambda: pd.Timedelta("4h"))
    # exchange size limits per symbol; when set, sizing follows the live bot exactly
    # (risk-based, raised to the exchange minimum, skipped above max_trade_risk_pct)
    rules: dict[str, SymbolRules] | None = None
    max_trade_risk_pct: float = 5.0


@dataclass
class Position:
    symbol: str
    side: int  # +1 long, -1 short
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    sl: float
    tp: float
    fees: float
    funding: float = 0.0
    bars_held: int = 0


@dataclass
class PendingEntry:
    side: int
    limit: float
    qty: float
    atr: float


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series  # marked to market at every bar close
    halted_at: pd.Timestamp | None
    limit_orders: int = 0
    limit_fills: int = 0


class Backtester:
    def __init__(
        self,
        bars: dict[str, pd.DataFrame],
        predictions: pd.DataFrame,
        params: BacktestParams,
        funding: dict[str, pd.DataFrame] | None = None,
        intrabar: dict[str, pd.DataFrame] | None = None,
    ):
        """bars: per symbol, stored klines (open_time, open, high, low, close).
        predictions: open_time index, columns symbol, score (-1..+1, the ensemble), atr."""
        self.p = params
        self.symbols = sorted(predictions["symbol"].unique())
        self.bars = {s: bars[s].set_index("open_time")[["open", "high", "low", "close"]]
                     for s in self.symbols}
        self.preds = {s: g.drop(columns="symbol") for s, g in predictions.groupby("symbol")}
        funding = funding or {}
        no_funding = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
        self.funding = {s: funding[s].set_index("funding_time")["funding_rate"].sort_index()
                        if s in funding and funding[s] is not None else no_funding
                        for s in self.symbols}
        intrabar = intrabar or {}
        self.resolvers = {s: IntrabarResolver(intrabar.get(s), params.bar) for s in self.symbols}

    # --- order execution -------------------------------------------------------------
    def _slip(self, price: float, side: int, opening: bool) -> float:
        """Market orders fill worse: buying higher, selling lower."""
        buy = (side == 1) == opening
        s = self.p.slippage_bps / 10_000
        return price * (1 + s) if buy else price * (1 - s)

    def _close(self, pos: Position, raw_price: float, time: pd.Timestamp, reason: str,
               maker: bool = False) -> float:
        price = raw_price if maker else self._slip(raw_price, pos.side, opening=False)
        exit_fee = pos.qty * price * (self.p.maker_fee if maker else self.p.fee)
        gross = pos.side * pos.qty * (price - pos.entry_price)
        pnl = gross - pos.fees - exit_fee - pos.funding
        self.trades.append({
            "symbol": pos.symbol, "side": "long" if pos.side == 1 else "short",
            "entry_time": pos.entry_time, "exit_time": time, "entry": pos.entry_price,
            "exit": price, "qty": pos.qty, "notional": pos.qty * pos.entry_price,
            "gross": gross, "fees": pos.fees + exit_fee, "funding": pos.funding, "pnl": pnl,
            "bars": pos.bars_held, "reason": reason,
        })
        self.cash += gross - exit_fee - pos.funding  # entry fee was paid at entry
        del self.positions[pos.symbol]
        return pnl

    def _take(self, pos: Position, time: pd.Timestamp) -> None:
        self._close(pos, pos.tp, time, "take", maker=self.p.take_profit_limit)

    def _check_barriers(self, pos: Position, t: pd.Timestamp, bar: pd.Series) -> None:
        long = pos.side == 1
        o, h, lo = bar["open"], bar["high"], bar["low"]
        # gap through a level at the open: filled at the open
        if (long and o <= pos.sl) or (not long and o >= pos.sl):
            self._close(pos, o, t, "stop")
            return
        if (long and o >= pos.tp) or (not long and o <= pos.tp):
            self._close(pos, o, t, "take", maker=self.p.take_profit_limit)
            return
        tp_hit = h >= pos.tp if long else lo <= pos.tp
        sl_hit = lo <= pos.sl if long else h >= pos.sl
        if tp_hit and sl_hit:
            tp_first = self.resolvers[pos.symbol].tp_first(self._ns(t), pos.tp, pos.sl, long)
            tp_hit, sl_hit = tp_first, not tp_first
        if sl_hit:
            self._close(pos, pos.sl, t + self.p.bar, "stop")
        elif tp_hit:
            self._take(pos, t + self.p.bar)

    def _fill_pending(self, s: str, order: PendingEntry, t: pd.Timestamp, bar: pd.Series) -> None:
        """Limit entry placed at the previous close: fill inside bar t or expire."""
        long = order.side == 1
        self.limit_orders += 1
        o = bar["open"]
        price = min(o, order.limit) if long else max(o, order.limit)  # gap -> better open
        sl = price - order.side * self.p.sl_atr * order.atr
        tp = price + order.side * self.p.tp_atr * order.atr
        resolver = self.resolvers[s]
        if resolver.has_data(self._ns(t)):
            filled, outcome = resolver.limit_fill(self._ns(t), order.limit, sl, tp, long)
        else:  # no intrabar data: conservative - a stop in the bar counts, a take does not
            level = limit_through(order.limit, long)
            filled = bar["low"] <= level if long else bar["high"] >= level
            stop_in_bar = bar["low"] <= sl if long else bar["high"] >= sl
            outcome = "stop" if stop_in_bar else None
        if not filled:
            return
        self.limit_fills += 1
        fee = order.qty * price * self.p.maker_fee
        self.cash -= fee
        pos = Position(s, order.side, t, price, order.qty, sl, tp, fee)
        self.positions[s] = pos
        if outcome == "stop":
            self._close(pos, sl, t + self.p.bar, "stop")
        elif outcome == "take":
            self._take(pos, t + self.p.bar)

    def _charge_funding(self, pos: Position, t: pd.Timestamp, close: float) -> None:
        rates = self.funding[pos.symbol]
        events = rates[(rates.index > t) & (rates.index <= t + self.p.bar)]
        for rate in events:
            pos.funding += pos.side * pos.qty * close * rate  # longs pay positive funding

    def _size(self, symbol: str, price: float, atr: float, equity: float) -> float:
        p = self.p
        if p.rules is None:
            qty = equity * p.risk_pct / 100 / (p.sl_atr * atr)
            return min(qty, equity * p.max_leverage / price)
        other = sum(pos.qty * pos.entry_price for s, pos in self.positions.items() if s != symbol)
        other += sum(o.qty * o.limit for s, o in self.pending.items() if s != symbol)
        return position_size(equity, price, p.sl_atr * atr, p.rules[symbol], p.risk_pct,
                             p.max_trade_risk_pct, p.max_leverage, other).qty

    def _enter(self, symbol: str, side: int, t_close: pd.Timestamp, close: float, atr: float,
               equity: float) -> None:
        if self.p.entry == "limit":
            limit = close - side * self.p.limit_offset_atr * atr
            qty = self._size(symbol, limit, atr, equity)
            if qty > 0:
                self.pending[symbol] = PendingEntry(side, limit, qty, atr)
            return
        price = self._slip(close, side, opening=True)
        qty = self._size(symbol, price, atr, equity)
        if qty <= 0:
            return
        fee = qty * price * self.p.fee
        self.cash -= fee
        self.positions[symbol] = Position(
            symbol, side, t_close, price, qty,
            sl=price - side * self.p.sl_atr * atr, tp=price + side * self.p.tp_atr * atr,
            fees=fee)

    @staticmethod
    def _ns(t: pd.Timestamp) -> int:
        return int(_to_ns(pd.Series([t]))[0])

    # --- main loop ---------------------------------------------------------------------
    def run(self) -> BacktestResult:
        p = self.p
        self.cash = p.initial_equity
        self.positions: dict[str, Position] = {}
        self.pending: dict[str, PendingEntry] = {}
        self.trades: list[dict] = []
        self.limit_orders = self.limit_fills = 0
        peak, halted_at = p.initial_equity, None
        day, day_start_equity = None, p.initial_equity
        equity_curve = {}

        times = sorted(set().union(*(g.index for g in self.preds.values())))
        for t in times:
            t_close = t + p.bar
            # 1. intrabar: pending limit entries, exchange-side stops/takes, funding
            filled_now = set()
            for s, order in list(self.pending.items()):
                del self.pending[s]
                if t in self.bars[s].index:
                    self._fill_pending(s, order, t, self.bars[s].loc[t])
                    filled_now.add(s)
            for s in list(self.positions):
                if t not in self.bars[s].index or s in filled_now:  # fill bar already checked
                    continue
                pos = self.positions[s]
                bar = self.bars[s].loc[t]
                self._charge_funding(pos, t, bar["close"])
                self._check_barriers(pos, t, bar)

            # 2. bar close: mark to market, risk limits
            closes = {s: self.bars[s].at[t, "close"] for s in self.symbols
                      if t in self.bars[s].index}
            equity = self._equity(closes)
            if t_close.date() != day:
                day, day_start_equity = t_close.date(), equity
            peak = max(peak, equity)
            if halted_at is None and equity < peak * (1 - p.max_dd_pct / 100):
                halted_at = t_close
            day_blocked = equity < day_start_equity * (1 - p.daily_loss_pct / 100)

            # 3. exits on time / opposite signal, then entries
            for s in self.symbols:
                if s not in closes or t not in self.preds[s].index:
                    continue
                row = self.preds[s].loc[t]
                close = closes[s]
                pos = self.positions.get(s)
                if pos is not None:
                    pos.bars_held += 1
                    if pos.bars_held >= p.horizon:
                        self._close(pos, close, t_close, "time")
                    elif exit_signal(pos.side, row["score"], p.exit_score):
                        self._close(pos, close, t_close, "signal")
                if s in self.positions or halted_at is not None or day_blocked:
                    continue
                side = entry_side(row["score"], p.enter_score)
                if side:
                    self._enter(s, side, t_close, close, row["atr"], self._equity(closes))
            equity_curve[t_close] = self._equity(closes)

        last = {s: self.bars[s]["close"].iloc[-1] for s in self.symbols}
        for s in list(self.positions):
            self._close(self.positions[s], last[s], self.bars[s].index[-1] + p.bar, "end")
        return BacktestResult(pd.DataFrame(self.trades), pd.Series(equity_curve, name="equity"),
                              halted_at, self.limit_orders, self.limit_fills)

    def _equity(self, closes: dict[str, float]) -> float:
        unrealized = sum(
            pos.side * pos.qty * (closes.get(s, pos.entry_price) - pos.entry_price) - pos.funding
            for s, pos in self.positions.items())
        return self.cash + unrealized


def run_backtest(
    bars: dict[str, pd.DataFrame], predictions: pd.DataFrame, params: BacktestParams,
    funding: dict[str, pd.DataFrame] | None = None,
    intrabar: dict[str, pd.DataFrame] | None = None,
) -> BacktestResult:
    return Backtester(bars, predictions, params, funding, intrabar).run()
