"""Exchange trading rules (tick/step/min size) and position sizing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal


@dataclass(frozen=True)
class SymbolRules:
    symbol: str
    tick: float  # price step
    step: float  # quantity step
    min_qty: float
    min_notional: float

    @classmethod
    def from_exchange_info(cls, info: dict, symbol: str) -> SymbolRules:
        s = next(x for x in info["symbols"] if x["symbol"] == symbol)
        f = {x["filterType"]: x for x in s["filters"]}
        return cls(
            symbol=symbol,
            tick=float(f["PRICE_FILTER"]["tickSize"]),
            step=float(f["LOT_SIZE"]["stepSize"]),
            min_qty=max(float(f["LOT_SIZE"]["minQty"]), float(f["MARKET_LOT_SIZE"]["minQty"])),
            min_notional=float(f["MIN_NOTIONAL"]["notional"]),
        )

    def round_price(self, price: float, up: bool = False) -> float:
        return _round_to(price, self.tick, up)

    def round_qty(self, qty: float, up: bool = False) -> float:
        return _round_to(qty, self.step, up)

    def fmt_price(self, price: float) -> str:
        return _fmt(price, self.tick)

    def fmt_qty(self, qty: float) -> str:
        return _fmt(qty, self.step)


def _round_to(x: float, step: float, up: bool) -> float:
    d = Decimal(str(x)) / Decimal(str(step))
    n = d.to_integral_value(ROUND_CEILING if up else ROUND_FLOOR)
    return float(n * Decimal(str(step)))


def _fmt(x: float, step: float) -> str:
    decimals = max(0, -Decimal(str(step)).normalize().as_tuple().exponent)
    return f"{x:.{decimals}f}"


@dataclass
class SizeDecision:
    qty: float  # 0 -> skip
    risk_pct: float  # of equity if the stop is hit (before fees)
    leverage: float
    reason: str


def position_size(
    equity: float, price: float, stop_distance: float, rules: SymbolRules,
    risk_pct: float, max_trade_risk_pct: float, max_leverage: float,
    other_notional: float = 0.0,
) -> SizeDecision:
    """Risk-based size, raised to the exchange minimum when needed (small accounts).
    Skips the trade when the minimum size would risk more than max_trade_risk_pct or when the
    total leverage (including other open positions) would exceed max_leverage."""
    if equity <= 0 or price <= 0 or stop_distance <= 0:
        return SizeDecision(0, 0, 0, "invalid inputs")
    qty = rules.round_qty(equity * risk_pct / 100 / stop_distance)
    min_qty = max(rules.min_qty, rules.round_qty(rules.min_notional / price, up=True))
    if qty < min_qty:
        qty = min_qty
    risk = 100 * qty * stop_distance / equity
    leverage = (qty * price + other_notional) / equity
    if risk > max_trade_risk_pct + 1e-9:
        return SizeDecision(0, risk, leverage,
                            f"minimum size risks {risk:.1f}% > {max_trade_risk_pct}%")
    if leverage > max_leverage + 1e-9:
        cap_qty = rules.round_qty((max_leverage * equity - other_notional) / price)
        if cap_qty < min_qty:
            return SizeDecision(0, risk, leverage,
                                f"leverage {leverage:.1f}x > {max_leverage}x at minimum size")
        qty = cap_qty
        risk = 100 * qty * stop_distance / equity
        leverage = (qty * price + other_notional) / equity
    return SizeDecision(qty, risk, leverage, "ok")


def required_leverage_setting(max_leverage: float) -> int:
    """Exchange leverage setting: whole number covering the planned maximum."""
    return max(1, math.ceil(max_leverage))
