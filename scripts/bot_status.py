"""Show the bot's state: equity, positions and orders, recent signals, trade journal.

    python scripts/bot_status.py            # demo or live, following execution.testnet
    python scripts/bot_status.py --reset-halt
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import pandas as pd

from biforex.config import load_settings
from biforex.trading.binance import BinanceFutures
from biforex.trading.state import State

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("--reset-halt", action="store_true",
                        help="allow new entries again after the max-drawdown halt")
    parser.add_argument("--signals", type=int, default=12, help="recent signals to show")
    args = parser.parse_args()

    settings = load_settings(args.config)
    demo = settings.execution.testnet
    mode = "demo" if demo else "live"
    path = settings.data.dir / "bot" / f"{mode}.sqlite"
    if not path.exists():
        sys.exit(f"no state yet: {path}")
    state = State(path)

    if args.reset_halt:
        state.set("halted", None)
        state.set("peak_equity", None)
        print("halt cleared; peak equity resets to the current equity on the next bar")
        return 0

    prefix = "BINANCE_TESTNET_API" if demo else "BINANCE_API"
    client = BinanceFutures(os.environ.get(f"{prefix}_KEY", ""),
                            os.environ.get(f"{prefix}_SECRET", ""), demo=demo)
    equity, available = client.usdt_equity()
    print(f"== {mode.upper()} | equity {equity:.4f} USDT, available {available:.4f} | "
          f"peak {state.get('peak_equity')} | halted {state.get('halted')} | "
          f"last bar {state.get('last_bar')}")
    for s in settings.symbols:
        pos = client.position(s)
        plan = state.plan(s)
        algos = [(a["orderType"], a["triggerPrice"]) for a in client.open_algo_orders(s)]
        orders = [(o["type"], o["side"], o["price"], o["origQty"]) for o in client.open_orders(s)]
        print(f"{s}: position {pos['qty']} @ {pos['entry']} (mark {pos['mark']:.2f}) | plan "
              f"{(plan.status, plan.side, plan.sl_price, plan.tp_price) if plan else None}\n"
              f"    stops {algos} | orders {orders}")

    db = sqlite3.connect(path)
    sig = pd.read_sql(f"SELECT * FROM signals ORDER BY bar DESC LIMIT {int(args.signals)}", db)
    print("\nrecent signals:\n", sig.round(3).to_string(index=False) if len(sig) else "none")
    trades = pd.DataFrame(state.trades())
    if trades.empty:
        print("\nno closed trades yet")
    else:
        print("\ntrades:\n", trades.drop(columns=["id"]).round(4).to_string(index=False))
        print(f"\ntotal pnl {trades['pnl'].sum():.4f} USDT over {len(trades)} trades, "
              f"win rate {100 * (trades['pnl'] > 0).mean():.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
