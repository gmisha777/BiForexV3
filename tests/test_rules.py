import pytest

from biforex.trading.rules import SymbolRules, position_size

BTC = SymbolRules("BTCUSDT", tick=0.1, step=0.001, min_qty=0.001, min_notional=50.0)
ETH = SymbolRules("ETHUSDT", tick=0.01, step=0.001, min_qty=0.001, min_notional=20.0)


def test_rounding_and_formatting():
    assert BTC.round_qty(0.0019) == 0.001
    assert BTC.round_qty(0.0011, up=True) == 0.002
    assert ETH.round_price(2686.576) == 2686.57
    assert ETH.round_price(2686.571, up=True) == 2686.58
    assert BTC.fmt_qty(0.001) == "0.001" and BTC.fmt_price(83458.2) == "83458.2"


def test_risk_based_size_on_a_large_account():
    d = position_size(5000, 2700, 40, ETH, risk_pct=1, max_trade_risk_pct=5, max_leverage=5)
    assert d.qty == pytest.approx(1.25)  # 50 USDT risk / 40 USDT stop
    assert d.risk_pct == pytest.approx(1.0)


def test_small_account_is_raised_to_exchange_minimum():
    # 20 USDT: 1% risk would be 0.0002 BTC; minimum is 0.001 BTC (~83 USDT)
    d = position_size(20, 83_000, 900, BTC, risk_pct=1, max_trade_risk_pct=5, max_leverage=5)
    assert d.qty == 0.001
    assert d.risk_pct == pytest.approx(4.5)
    assert d.leverage == pytest.approx(4.15)


def test_eth_minimum_follows_min_notional():
    d = position_size(20, 2700, 38, ETH, risk_pct=1, max_trade_risk_pct=5, max_leverage=5)
    assert d.qty == 0.008  # ceil(20 / 2700, step) -> 0.008 ETH = 21.6 USDT


def test_skip_when_minimum_risks_too_much():
    d = position_size(20, 83_000, 1500, BTC, risk_pct=1, max_trade_risk_pct=5, max_leverage=5)
    assert d.qty == 0 and "risks" in d.reason


def test_skip_when_leverage_including_other_positions_too_high():
    # BTC minimum (83 USDT) + open ETH 21.6 USDT on 20 USDT = 5.2x > 5x
    d = position_size(20, 83_000, 900, BTC, 1, 5, max_leverage=5, other_notional=21.6)
    assert d.qty == 0 and "leverage" in d.reason


def test_large_account_capped_by_leverage():
    d = position_size(1000, 2700, 5, ETH, risk_pct=1, max_trade_risk_pct=5, max_leverage=3)
    assert d.qty == pytest.approx(1.111)  # 3000 / 2700, floored to step
    assert d.leverage <= 3
