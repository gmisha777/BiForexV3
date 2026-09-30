import pandas as pd
import pytest

from biforex.config import Settings
from biforex.trading import bot as bot_module
from biforex.trading.bot import Bot
from biforex.trading.live_data import Decision
from biforex.trading.state import Plan, State

BAR = pd.Timestamp("2026-09-28 08:00", tz="UTC")
INFO = {"symbols": [
    {"symbol": "BTCUSDT", "filters": [
        {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
        {"filterType": "MARKET_LOT_SIZE", "minQty": "0.001"},
        {"filterType": "MIN_NOTIONAL", "notional": "50"}]},
    {"symbol": "ETHUSDT", "filters": [
        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
        {"filterType": "MARKET_LOT_SIZE", "minQty": "0.001"},
        {"filterType": "MIN_NOTIONAL", "notional": "20"}]},
]}


class FakeClient:
    demo = True

    def __init__(self, equity=20.0):
        self.equity = equity
        self.positions = {"BTCUSDT": 0.0, "ETHUSDT": 0.0}
        self.orders: list[dict] = []
        self.market_closes: list[str] = []

    def exchange_info(self):
        return INFO

    def usdt_equity(self):
        return self.equity, self.equity

    def position(self, s):
        return {"qty": self.positions[s], "entry": 2700.0, "mark": 2700.0}

    def new_order(self, **p):
        self.orders.append(p)
        if p["type"] == "MARKET":
            self.market_closes.append(p["symbol"])
            self.positions[p["symbol"]] = 0.0
        return {"status": "NEW"}

    def open_algo_orders(self, s):
        return []

    def open_orders(self, s):
        return []

    def cancel(self, s, cid):
        pass

    def cancel_algo(self, cid):
        pass

    def order(self, s, cid):
        return {"status": "NEW", "executedQty": "0"}

    def user_trades(self, s, start):
        return []

    def income(self, s, start, t=None):
        return []


class FakeScorer:
    def __init__(self, scores):
        self.scores = scores  # symbol -> ensemble score

    def decide(self, s, bar):
        close = 83_000.0 if s == "BTCUSDT" else 2700.0
        atr = 0.011 * close
        return Decision(bar, self.scores[s], close=close, atr=atr, parts={"x": self.scores[s]})


@pytest.fixture
def make_bot(tmp_path, monkeypatch):
    monkeypatch.setattr(bot_module, "refresh_data", lambda *a, **k: None)
    monkeypatch.setattr(bot_module, "STOP_FILE", tmp_path / "STOP")

    def make(probs, equity=20.0, notifier=None):
        s = Settings(symbols=["BTCUSDT", "ETHUSDT"])
        client = FakeClient(equity)
        b = Bot(s, client, State(tmp_path / "state.sqlite"), FakeScorer(probs),
                notifier=notifier)
        return b, client
    return make


NONE = 0.0
UP = 0.18  # above enter_score 0.10


def test_entries_use_exchange_minimum_and_limit_below_close(make_bot):
    b, c = make_bot({"BTCUSDT": UP, "ETHUSDT": UP}, equity=25.0)
    b.on_bar_close(BAR)
    entries = {o["symbol"]: o for o in c.orders}
    assert entries["BTCUSDT"]["quantity"] == "0.001"  # 3.7% risk on 25 USDT, allowed (<5%)
    assert entries["ETHUSDT"]["quantity"] == "0.008"  # 20 USDT min notional / 2700
    assert float(entries["ETHUSDT"]["price"]) == pytest.approx(2700 - 0.1 * 29.7, abs=0.01)
    assert b.state.plan("BTCUSDT").status == "pending"


def test_second_minimum_order_skipped_when_leverage_too_high(make_bot):
    b, c = make_bot({"BTCUSDT": UP, "ETHUSDT": UP}, equity=20.0)
    b.on_bar_close(BAR)
    # BTC 0.001 (~83 USDT) is 4.2x on 20 USDT; adding ETH (~21.6 USDT) makes 5.2x > 5x
    assert [o["symbol"] for o in c.orders] == ["BTCUSDT"]


def test_minimum_order_skipped_when_it_risks_too_much(make_bot):
    b, c = make_bot({"BTCUSDT": UP, "ETHUSDT": NONE}, equity=18.0)
    b.on_bar_close(BAR)
    # BTC stop 1 ATR = 913 USDT per BTC -> 0.913 USDT on 0.001 = 5.1% of 18 USDT > 5%
    assert c.orders == []


def test_no_signal_no_orders(make_bot):
    b, c = make_bot({"BTCUSDT": NONE, "ETHUSDT": NONE})
    b.on_bar_close(BAR)
    assert c.orders == []


def test_stop_file_blocks_entries(make_bot, tmp_path):
    (tmp_path / "STOP").touch()
    b, c = make_bot({"BTCUSDT": UP, "ETHUSDT": UP})
    b.on_bar_close(BAR)
    assert c.orders == []


def test_drawdown_halt_blocks_entries(make_bot):
    b, c = make_bot({"BTCUSDT": UP, "ETHUSDT": UP}, equity=13.0)
    b.state.set("peak_equity", 20.0)  # -35% > 30% limit
    b.on_bar_close(BAR)
    assert c.orders == [] and b.state.get("halted")


def test_time_exit_after_horizon(make_bot):
    b, c = make_bot({"BTCUSDT": NONE, "ETHUSDT": NONE})
    c.positions["ETHUSDT"] = 0.008
    signal_bar = BAR - 12 * pd.Timedelta("4h")
    b.state.save_plan(Plan("ETHUSDT", "open", 1, signal_bar.isoformat(), 30.0, "bfe1", 0, 0.008,
                           2700.0, signal_bar.isoformat(), "bfs1", 2670.0, "bfp1", 2760.0))
    b.on_bar_close(BAR)
    assert c.market_closes == ["ETHUSDT"]
    assert b.state.plan("ETHUSDT") is None
    assert b.state.trades()[-1]["reason"] == "time"


def test_last_closed_bar_waits_for_delay(make_bot):
    b, _ = make_bot({"BTCUSDT": NONE, "ETHUSDT": NONE})
    assert b.last_closed_bar(pd.Timestamp("2026-09-28 12:00:10", tz="UTC")) == BAR - pd.Timedelta(
        "4h")
    assert b.last_closed_bar(pd.Timestamp("2026-09-28 12:00:30", tz="UTC")) == BAR


def test_late_bar_opens_nothing(make_bot):
    b, c = make_bot({"BTCUSDT": UP, "ETHUSDT": UP}, equity=25.0)
    b.step(now=BAR + pd.Timedelta("4h") + pd.Timedelta("2h"))  # bot started 2 h after the close
    assert c.orders == []
    b.state.set("last_bar", None)
    b.step(now=BAR + pd.Timedelta("4h") + pd.Timedelta("1min"))  # on time
    assert {o["symbol"] for o in c.orders} == {"BTCUSDT", "ETHUSDT"}


class CaptureNotifier:
    enabled = True

    def __init__(self):
        self.messages = []

    def send(self, text, key=None, every_s=0):
        self.messages.append(text)


def test_notifications_for_closed_trade_and_halt(make_bot):
    n = CaptureNotifier()
    b, c = make_bot({"BTCUSDT": NONE, "ETHUSDT": NONE}, equity=13.0, notifier=n)
    b.state.set("peak_equity", 20.0)
    c.positions["ETHUSDT"] = 0.008
    signal_bar = BAR - 12 * pd.Timedelta("4h")
    b.state.save_plan(Plan("ETHUSDT", "open", 1, signal_bar.isoformat(), 30.0, "bfe1", 0, 0.008,
                           2700.0, signal_bar.isoformat(), "bfs1", 2670.0, "bfp1", 2760.0))
    b.on_bar_close(BAR)
    assert any("ЗУПИНЕНО" in m for m in n.messages)
    assert any("ETHUSDT LONG закрито (ліміт часу)" in m for m in n.messages)
