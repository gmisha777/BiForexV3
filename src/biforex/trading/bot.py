"""Trading bot: same rules as the backtest, executed on Binance USDT-M futures.

Every poll (30 s) the bot syncs each symbol with the exchange:
  pending entry filled  -> position becomes "open", stop + take profit are placed;
  open position         -> stop (algo STOP_MARKET, closePosition) and take profit (LIMIT
                           reduceOnly) must exist, otherwise they are (re)placed;
  position gone         -> leftover orders cancelled, trade written to the journal.
After each bar close (+ delay) it refreshes data, predicts, applies time / signal exits,
risk limits, and places new limit entries valid for one bar.
"""

from __future__ import annotations

import logging
import time
import uuid

import pandas as pd
import requests

from ..config import PROJECT_ROOT, Settings
from ..data.schema import timeframe_delta
from ..notify import Notifier, net_error
from .binance import BinanceError, BinanceFutures
from .live_data import Prediction, Predictor, refresh_data
from .rules import SymbolRules, position_size
from .state import Plan, State

log = logging.getLogger(__name__)
STOP_FILE = PROJECT_ROOT / "STOP"  # create this file to block new entries
# a bar processed later than this after its close (bot started or was down) opens no positions:
# the backtest enters right at the close, not hours later
MAX_ENTRY_LATENESS = pd.Timedelta("15min")
DAILY_REPORT_UTC_HOUR = 6  # 09:00 Kyiv in summer, 08:00 in winter
REASONS = {"take": "тейк", "stop": "стоп", "time": "ліміт часу",
           "signal": "протилежний сигнал", "no-stop": "стоп неможливо поставити"}


def _local(ts: pd.Timestamp) -> pd.Timestamp:
    """Kyiv time for messages (falls back to UTC if the time zone database is missing)."""
    try:
        return ts.tz_convert("Europe/Kyiv")
    except Exception:
        return ts


def _ms(ts: str | pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).timestamp() * 1000)


class Bot:
    def __init__(self, settings: Settings, client: BinanceFutures, state: State,
                 predictor: Predictor, dry_run: bool = False, notifier: Notifier | None = None):
        self.s = settings
        self.client = client
        self.state = state
        self.predictor = predictor
        self.dry_run = dry_run
        self.notify = notifier or Notifier()
        self.symbols = settings.symbols
        self.bar = timeframe_delta(settings.timeframe)
        info = client.exchange_info()
        self.rules = {s: SymbolRules.from_exchange_info(info, s) for s in self.symbols}
        self.session = requests.Session()
        self.forced: dict[str, int] = {}  # demo testing: symbol -> side for the next bar

    # --- setup -------------------------------------------------------------------------
    def setup(self) -> None:
        if not self.client.position_mode_is_one_way():
            raise RuntimeError("account is in hedge mode; switch to one-way position mode")
        if self.dry_run:
            return
        for s in self.symbols:
            self.client.set_isolated(s)
            self.client.set_leverage(s, self.s.execution.exchange_leverage)
        log.info("isolated margin, leverage %dx set for %s", self.s.execution.exchange_leverage,
                 self.symbols)

    @staticmethod
    def _cid(kind: str) -> str:
        return f"bf{kind}{uuid.uuid4().hex[:20]}"

    # --- main loop -----------------------------------------------------------------------
    def last_closed_bar(self, now: pd.Timestamp) -> pd.Timestamp:
        ready = now - pd.Timedelta(seconds=self.s.bot.bar_close_delay_seconds)
        return ready.floor(self.bar) - self.bar

    def step(self, now: pd.Timestamp | None = None) -> None:
        now = now or pd.Timestamp.now(tz="UTC")
        for s in self.symbols:
            self.sync(s)
        bar = self.last_closed_bar(now)
        if self.state.get("last_bar") != bar.isoformat():
            late = now - (bar + self.bar) > MAX_ENTRY_LATENESS
            self.on_bar_close(bar, allow_entries=not late)
            self.state.set("last_bar", bar.isoformat())

    def run_forever(self) -> None:
        log.info("bot started (%s, %s)", "DEMO" if self.client.demo else "LIVE ACCOUNT",
                 "dry run" if self.dry_run else "trading")
        equity, _ = self.client.usdt_equity()
        self.notify.send(f"▶️ Бот запущено ({'demo' if self.client.demo else 'РЕАЛЬНИЙ рахунок'}), "
                         f"капітал {equity:.2f} USDT, {', '.join(self.symbols)}")
        self.started_at = pd.Timestamp.now(tz="UTC")
        failing_since = None
        try:
            while True:
                try:
                    self.step()
                    if failing_since is not None:
                        self.notify.send(f"✅ Працює знову після помилок з "
                                         f"{_local(failing_since):%H:%M}")
                        failing_since = None
                except requests.RequestException as exc:  # network: one line, no traceback
                    log.warning("network problem (%s); retrying next poll", net_error(exc))
                    if failing_since is None:
                        failing_since = pd.Timestamp.now(tz="UTC")
                    self.notify.send(f"❌ Немає зв'язку з біржею ({net_error(exc)}); бот повторює "
                                     "спроби. Стоп і тейк на біржі діють.",
                                     key="step-error", every_s=3600)
                except Exception as exc:
                    log.exception("step failed; retrying next poll")
                    if failing_since is None:
                        failing_since = pd.Timestamp.now(tz="UTC")
                    self.notify.send(f"❌ Помилка бота (повтори приглушено на 1 год): {exc}",
                                     key="step-error", every_s=3600)
                self._telegram()
                time.sleep(self.s.bot.poll_seconds)
        except KeyboardInterrupt:
            log.info("bot stopped by user")
            self.notify.send("⏹ Бота зупинено (Ctrl+C). Відкриті позиції зберігають стоп і тейк "
                             "на біржі.")

    # --- telegram: commands and daily report ---------------------------------------------
    def _telegram(self) -> None:
        if not self.notify.enabled:
            return
        try:
            for cmd in self.notify.commands():
                if cmd == "/status":
                    self.notify.send(self.status_text())
                elif cmd == "/report":
                    self.notify.send(self.report_text())
                else:
                    self.notify.send("Команди:\n/status — чи живий бот, капітал, позиції"
                                     "\n/report — угоди за 24 год")
            now = pd.Timestamp.now(tz="UTC")
            today = str(now.date())
            if now.hour >= DAILY_REPORT_UTC_HOUR and self.state.get("report_day") != today:
                self.state.set("report_day", today)
                self.notify.send(self.report_text())
        except Exception:
            log.exception("telegram handling failed")

    def status_text(self) -> str:
        now = pd.Timestamp.now(tz="UTC")
        equity, available = self.client.usdt_equity()
        mode = "demo" if self.client.demo else "LIVE"
        up = now - getattr(self, "started_at", now)
        nxt = now.floor(self.bar) + self.bar + pd.Timedelta(
            seconds=self.s.bot.bar_close_delay_seconds)
        lines = [f"🟢 Бот працює ({mode}) {int(up.total_seconds() // 3600)} год "
                 f"{int(up.total_seconds() % 3600 // 60)} хв",
                 f"Капітал {equity:.2f} USDT (вільно {available:.2f})",
                 f"Наступне рішення о {_local(nxt):%H:%M}"]
        for s in self.symbols:
            plan, pos = self.state.plan(s), self.client.position(s)
            if pos["qty"]:
                side = "LONG" if pos["qty"] > 0 else "SHORT"
                upnl = pos["qty"] * (pos["mark"] - pos["entry"])
                sl_tp = f" · SL {plan.sl_price:g} · TP {plan.tp_price:g}" if plan else ""
                lines.append(f"{s}: {side} {abs(pos['qty']):g} @ {pos['entry']:g}{sl_tp} · "
                             f"uPnL {upnl:+.3f}")
            elif plan and plan.status == "pending":
                side = "LONG" if plan.side == 1 else "SHORT"
                lines.append(f"{s}: лімітка {side} {plan.qty:g} @ {plan.entry_limit:g} чекає")
            else:
                lines.append(f"{s}: позиції немає")
        last = self.state.last_signals(len(self.symbols))
        if last:
            bar = pd.Timestamp(last[0]["bar"]) + self.bar
            probs = ", ".join(f"{r['symbol'][:3]} ↑{r['p_up']:.2f} ↓{r['p_down']:.2f}"
                              for r in last)
            lines.append(f"Останній сигнал {_local(bar):%d.%m %H:%M}: {probs} "
                         f"(вхід від {self.s.signal.enter_prob:g})")
        if self.state.get("halted"):
            lines.append("⛔ Нові входи ЗУПИНЕНО (макс. просадка)")
        if STOP_FILE.exists():
            lines.append("⏸ Є файл STOP: нових входів немає")
        return "\n".join(lines)

    def report_text(self) -> str:
        since = pd.Timestamp.now(tz="UTC") - pd.Timedelta("24h")
        trades = self.state.trades()
        day = [t for t in trades if pd.Timestamp(t["exit_time"]) >= since]
        lines = ["📊 Щоденний звіт", self.status_text(), ""]
        if day:
            lines.append(f"За 24 год: угод {len(day)}, PnL {sum(t['pnl'] for t in day):+.4f}")
            lines += [f"  {t['symbol']} {t['side']} {t['reason']} {t['pnl']:+.4f}" for t in day]
        else:
            lines.append("За 24 год закритих угод немає")
        if trades:
            total = sum(t["pnl"] for t in trades)
            wins = sum(t["pnl"] > 0 for t in trades)
            lines.append(f"Усього: угод {len(trades)}, прибуткових {wins}, PnL {total:+.4f} USDT")
        return "\n".join(lines)

    # --- sync with the exchange ----------------------------------------------------------
    def sync(self, s: str) -> None:
        plan = self.state.plan(s)
        pos = self.client.position(s)
        if plan and plan.status == "pending":
            plan = self._check_entry(s, plan, pos)
        if pos["qty"] != 0:
            if plan is None or plan.status != "open":
                plan = self._adopt(s, pos, plan)
            self._protect(s, plan, pos)
        elif plan and plan.status == "open":
            self._finish(s, plan)

    def _check_entry(self, s: str, plan: Plan, pos: dict) -> Plan | None:
        try:
            order = self.client.order(s, plan.entry_cid)
        except BinanceError as exc:
            if exc.code == -2013:  # order does not exist: placement failed
                log.warning("%s: entry order %s not found, dropping plan", s, plan.entry_cid)
                self.state.delete_plan(s)
                return None
            raise
        filled = float(order["executedQty"])
        if filled > 0 and pos["qty"] != 0:
            if order["status"] == "PARTIALLY_FILLED":
                self.client.cancel(s, plan.entry_cid)  # keep what we got, protect it now
            plan.status = "open"
            plan.entry_price = pos["entry"]
            plan.qty = abs(pos["qty"])
            plan.entry_time = pd.Timestamp.now(tz="UTC").isoformat()
            self._set_levels(s, plan)
            self.state.save_plan(plan)
            log.info("%s: entry filled %s %.4f @ %.2f | SL %.2f TP %.2f", s,
                     "LONG" if plan.side == 1 else "SHORT", plan.qty, plan.entry_price,
                     plan.sl_price, plan.tp_price)
            return plan
        if order["status"] in ("CANCELED", "EXPIRED", "REJECTED") and filled == 0:
            log.info("%s: entry order %s %s, not filled", s, plan.entry_cid, order["status"])
            self.state.delete_plan(s)
            return None
        return plan

    def _set_levels(self, s: str, plan: Plan) -> None:
        r, lc = self.rules[s], self.s.label
        long = plan.side == 1
        plan.sl_price = r.round_price(plan.entry_price - plan.side * lc.sl_atr * plan.atr,
                                      up=not long)
        plan.tp_price = r.round_price(plan.entry_price + plan.side * lc.tp_atr * plan.atr,
                                      up=not long)

    def _adopt(self, s: str, pos: dict, plan: Plan | None) -> Plan:
        """A position the bot has no open plan for (e.g. restart after a crash): protect it."""
        atr = self.state.get(f"last_atr:{s}") or 0.015 * pos["entry"]
        side = 1 if pos["qty"] > 0 else -1
        log.warning("%s: position %.4f without an open plan; adopting with ATR %.2f", s,
                    pos["qty"], atr)
        new = Plan(symbol=s, status="open", side=side,
                   signal_bar=plan.signal_bar if plan else self.state.get("last_bar", ""),
                   atr=plan.atr if plan else atr, entry_cid=plan.entry_cid if plan else "",
                   entry_limit=0.0, qty=abs(pos["qty"]), entry_price=pos["entry"],
                   entry_time=pd.Timestamp.now(tz="UTC").isoformat())
        self._set_levels(s, new)
        self.state.save_plan(new)
        return new

    def _protect(self, s: str, plan: Plan, pos: dict) -> None:
        if self.dry_run:
            return
        r = self.rules[s]
        qty = abs(pos["qty"])
        close_side = "SELL" if plan.side == 1 else "BUY"
        algos = {a["clientAlgoId"] for a in self.client.open_algo_orders(s)}
        if plan.sl_cid not in algos:
            plan.sl_cid = self._cid("s")
            self.state.save_plan(plan)
            try:
                self.client.new_algo_order(
                    symbol=s, side=close_side, type="STOP_MARKET",
                    triggerPrice=r.fmt_price(plan.sl_price), closePosition="true",
                    workingType="CONTRACT_PRICE", clientAlgoId=plan.sl_cid)
                log.info("%s: stop placed at %.2f", s, plan.sl_price)
            except BinanceError as exc:
                log.error("%s: cannot place stop (%s); closing position at market", s, exc)
                self.notify.send(f"⚠️ {s}: біржа відхилила стоп ({exc.msg}); позицію закрито "
                                 "за ринком")
                self.close_market(s, "stop" if exc.code == -2021 else "no-stop")
                return
        orders = {o["clientOrderId"]: o for o in self.client.open_orders(s)}
        tp = orders.get(plan.tp_cid)
        if tp is None or abs(float(tp["origQty"]) - float(tp["executedQty"]) - qty) > r.step / 2:
            if tp is not None:
                self.client.cancel(s, plan.tp_cid)
            plan.tp_cid = self._cid("p")
            self.state.save_plan(plan)
            try:
                self.client.new_order(
                    symbol=s, side=close_side, type="LIMIT", timeInForce="GTC",
                    quantity=r.fmt_qty(qty), price=r.fmt_price(plan.tp_price),
                    reduceOnly="true", newClientOrderId=plan.tp_cid)
                log.info("%s: take profit placed at %.2f for %.4f", s, plan.tp_price, qty)
            except BinanceError as exc:
                log.error("%s: cannot place take profit: %s (stop stays in place)", s, exc)
                self.notify.send(f"⚠️ {s}: біржа відхилила тейк ({exc.msg}); стоп на місці",
                                 key=f"tp-error:{s}", every_s=3600)

    def _finish(self, s: str, plan: Plan) -> None:
        """Position is flat: clean up, journal the trade."""
        for a in self.client.open_algo_orders(s):
            if a["clientAlgoId"].startswith("bf"):
                self.client.cancel_algo(a["clientAlgoId"])
        for o in self.client.open_orders(s):
            if o["clientOrderId"].startswith("bf"):
                self.client.cancel(s, o["clientOrderId"])
        reason = plan.exit_reason
        if not reason:
            try:
                tp = self.client.order(s, plan.tp_cid) if plan.tp_cid else None
            except BinanceError:
                tp = None
            reason = "take" if tp and tp["status"] == "FILLED" else "stop"
        # the entry order was placed at the signal bar close; fills cannot be earlier
        placed = pd.Timestamp(plan.signal_bar) + self.bar if plan.signal_bar else None
        start = _ms(placed) - 60_000 if placed is not None else _ms(plan.entry_time) - 3_600_000
        start = max(start, self.state.get(f"last_exit_ms:{s}", 0) + 1)  # not the previous trade
        fills = self.client.user_trades(s, start)
        close_side = "SELL" if plan.side == 1 else "BUY"
        exits = [t for t in fills if t["side"] == close_side]
        exit_qty = sum(float(t["qty"]) for t in exits)
        exit_price = (sum(float(t["qty"]) * float(t["price"]) for t in exits) / exit_qty
                      if exit_qty else 0.0)
        realized = sum(float(t["realizedPnl"]) for t in fills)
        fees = sum(float(t["commission"]) for t in fills if t["commissionAsset"] == "USDT")
        funding = -sum(float(i["income"]) for i in self.client.income(s, start, "FUNDING_FEE"))
        pnl = realized - fees - funding
        self.state.add_trade(
            symbol=s, side="long" if plan.side == 1 else "short", entry_time=plan.entry_time,
            exit_time=pd.Timestamp.now(tz="UTC").isoformat(), entry=plan.entry_price,
            exit=exit_price, qty=plan.qty, realized=realized, fees=fees, funding=funding,
            pnl=pnl, reason=reason)
        self.state.delete_plan(s)
        if fills:
            self.state.set(f"last_exit_ms:{s}", max(int(t["time"]) for t in fills))
        log.info("%s: trade closed (%s) entry %.2f exit %.2f qty %.4f | pnl %.4f USDT "
                 "(fees %.4f, funding %.4f)", s, reason, plan.entry_price, exit_price, plan.qty,
                 pnl, fees, funding)
        try:
            equity = self.client.usdt_equity()[0]
        except BinanceError:
            equity = float("nan")
        icon = "💰" if pnl > 0 else "🔻"
        side_txt = "LONG" if plan.side == 1 else "SHORT"
        self.notify.send(
            f"{icon} {s} {side_txt} закрито ({REASONS.get(reason, reason)})\n"
            f"{plan.entry_price:.2f} → {exit_price:.2f}, к-сть {plan.qty:g}\n"
            f"PnL {pnl:+.4f} USDT (комісії {fees:.4f}) · капітал {equity:.2f} USDT")

    def close_market(self, s: str, reason: str) -> None:
        plan = self.state.plan(s)
        pos = self.client.position(s)
        if plan:
            plan.exit_reason = reason
            self.state.save_plan(plan)
        if pos["qty"] == 0 or self.dry_run:
            log.info("%s: close (%s) %s", s, reason, "skipped in dry run" if self.dry_run
                     else "not needed, flat")
            return
        if plan and plan.tp_cid:
            self.client.cancel(s, plan.tp_cid)
        self.client.new_order(
            symbol=s, side="SELL" if pos["qty"] > 0 else "BUY", type="MARKET",
            quantity=self.rules[s].fmt_qty(abs(pos["qty"])), reduceOnly="true",
            newClientOrderId=self._cid("x"))
        log.info("%s: closed at market (%s)", s, reason)
        self.sync(s)

    # --- bar close -----------------------------------------------------------------------
    def on_bar_close(self, bar: pd.Timestamp, allow_entries: bool = True) -> None:
        log.info("=== bar %s closed ===%s", bar,
                 "" if allow_entries else " (processed late: exits only, no new entries)")
        # 1. unfilled entries live for one bar only
        for s in self.symbols:
            plan = self.state.plan(s)
            if plan and plan.status == "pending" and not self.dry_run:
                self.client.cancel(s, plan.entry_cid)
                self.sync(s)

        # 2. data and predictions
        fc = self.s.features.context_symbol
        refresh_data(self.s, sorted(set(self.symbols) | {fc}), self.session)
        preds: dict[str, Prediction | None] = {}
        for s in self.symbols:
            preds[s] = self.predictor.predict(s, bar)
            if preds[s]:
                self.state.set(f"last_atr:{s}", preds[s].atr)

        # 3. risk state
        equity, available = self.client.usdt_equity()
        blocked = self._risk_block(bar, equity)
        log.info("equity %.4f USDT (available %.4f)%s", equity, available,
                 f" | new entries blocked: {blocked}" if blocked else "")

        # 4. exits: time limit, strong opposite signal
        horizon, sc = self.s.label.horizon_bars, self.s.signal
        for s in self.symbols:
            plan, p = self.state.plan(s), preds[s]
            if not plan or plan.status != "open":
                continue
            held = int((bar - pd.Timestamp(plan.signal_bar)) / self.bar)
            opposite = (p.p_down if plan.side == 1 else p.p_up) if p else 0.0
            if held >= horizon:
                self.close_market(s, "time")
            elif opposite >= sc.exit_prob:
                self.close_market(s, "signal")

        # 5. entries
        for s in self.symbols:
            p = preds[s]
            if p is None:
                continue
            side = self.forced.pop(s, None) or self._signal(p)
            action = {1: "long", -1: "short", 0: "none"}[side]
            if side == 0:
                self.state.add_signal(bar.isoformat(), s, p.probs, action)
                continue
            if self.state.plan(s):
                self.state.add_signal(bar.isoformat(), s, p.probs, action, "position open")
                continue
            if not allow_entries:
                blocked = blocked or "bar processed late"
            if blocked:
                self.state.add_signal(bar.isoformat(), s, p.probs, action, f"blocked: {blocked}")
                log.info("%s: %s signal skipped (%s)", s, action, blocked)
                continue
            detail = self._enter(s, side, p, bar, equity)
            self.state.add_signal(bar.isoformat(), s, p.probs, action, detail)

    def _risk_block(self, bar: pd.Timestamp, equity: float) -> str:
        rc = self.s.risk
        peak = max(self.state.get("peak_equity") or equity, equity)
        self.state.set("peak_equity", peak)
        day = str((bar + self.bar).date())
        if self.state.get("day") != day:
            self.state.set("day", day)
            self.state.set("day_start_equity", equity)
        if not self.state.get("halted") and equity < peak * (1 - rc.max_dd_pct / 100):
            self.state.set("halted", bar.isoformat())
            log.critical("drawdown %.1f%% from peak %.2f: new entries halted until reset",
                         100 * (1 - equity / peak), peak)
            self.notify.send(f"⛔ Просадка {100 * (1 - equity / peak):.1f}% від максимуму "
                             f"{peak:.2f} USDT: нові входи ЗУПИНЕНО. Відкриті позиції зберігають "
                             "стопи. Відновити: scripts/bot_status.py --reset-halt")
        if self.state.get("halted"):
            return f"max drawdown (since {self.state.get('halted')})"
        if equity < self.state.get("day_start_equity", equity) * (1 - rc.daily_loss_pct / 100):
            if self.state.get("daily_limit_notified") != day:
                self.state.set("daily_limit_notified", day)
                self.notify.send(f"⏸ Досягнуто денний ліміт збитку {rc.daily_loss_pct:g}% "
                                 f"(капітал {equity:.2f} USDT): нових входів немає "
                                 "до 03:00 за Києвом")
            return "daily loss limit"
        if STOP_FILE.exists():
            return "STOP file present"
        return ""

    def _signal(self, p: Prediction) -> int:
        th = self.s.signal.enter_prob
        if p.p_up >= th and p.p_up > p.p_down:
            return 1
        if p.p_down >= th and p.p_down > p.p_up:
            return -1
        return 0

    def _enter(self, s: str, side: int, p: Prediction, bar: pd.Timestamp, equity: float) -> str:
        r, ex, rc = self.rules[s], self.s.execution, self.s.risk
        limit = p.close - side * ex.limit_offset_atr * p.atr
        limit = r.round_price(limit, up=side == -1)
        other = 0.0
        for o in self.symbols:
            if o == s:
                continue
            op = self.state.plan(o)
            if op:
                other += op.qty * (op.entry_price or op.entry_limit)
        size = position_size(equity, limit, self.s.label.sl_atr * p.atr, r, rc.per_trade_pct,
                             rc.max_trade_risk_pct, rc.max_leverage, other)
        side_txt = "LONG" if side == 1 else "SHORT"
        if size.qty == 0:
            log.info("%s: %s signal skipped: %s", s, side_txt, size.reason)
            return f"skipped: {size.reason}"
        plan = Plan(symbol=s, status="pending", side=side, signal_bar=bar.isoformat(),
                    atr=p.atr, entry_cid=self._cid("e"), entry_limit=limit, qty=size.qty)
        log.info("%s: %s signal -> limit %.2f qty %s (risk %.1f%%, leverage %.1fx)", s,
                 side_txt, limit, r.fmt_qty(size.qty), size.risk_pct, size.leverage)
        if self.dry_run:
            return "dry run"
        self.state.save_plan(plan)
        try:
            self.client.new_order(
                symbol=s, side="BUY" if side == 1 else "SELL", type="LIMIT", timeInForce="GTC",
                quantity=r.fmt_qty(size.qty), price=r.fmt_price(limit),
                newClientOrderId=plan.entry_cid)
        except BinanceError as exc:
            log.error("%s: entry order rejected: %s", s, exc)
            self.notify.send(f"⚠️ {s}: біржа відхилила вхідний ордер ({exc.msg})",
                             key=f"entry-error:{s}", every_s=3600)
            self.state.delete_plan(s)
            return f"rejected: {exc}"
        return f"limit {limit} qty {size.qty}"
