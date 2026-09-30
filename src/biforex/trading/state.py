"""Bot state in SQLite: position plans, trade journal, signals, risk counters.

A "plan" is the bot's intent for one symbol: a pending limit entry, then an open position
with its stop and take-profit orders. The exchange is the source of truth for quantities;
the plan remembers what the exchange does not (ATR at signal, signal bar, exit reason).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, fields
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS plans (
    symbol TEXT PRIMARY KEY, status TEXT, side INTEGER, signal_bar TEXT, atr REAL,
    entry_cid TEXT, entry_limit REAL, qty REAL, entry_price REAL, entry_time TEXT,
    sl_cid TEXT, sl_price REAL, tp_cid TEXT, tp_price REAL, exit_reason TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side TEXT, entry_time TEXT,
    exit_time TEXT, entry REAL, exit REAL, qty REAL, realized REAL, fees REAL, funding REAL,
    pnl REAL, reason TEXT
);
CREATE TABLE IF NOT EXISTS signals (
    bar TEXT, symbol TEXT, p_down REAL, p_flat REAL, p_up REAL, action TEXT, detail TEXT,
    PRIMARY KEY (bar, symbol)
);
"""


@dataclass
class Plan:
    symbol: str
    status: str  # "pending" (entry order working) | "open" (position held)
    side: int  # +1 long, -1 short
    signal_bar: str  # open time of the bar whose close produced the signal (ISO)
    atr: float
    entry_cid: str
    entry_limit: float
    qty: float
    entry_price: float = 0.0
    entry_time: str = ""
    sl_cid: str = ""
    sl_price: float = 0.0
    tp_cid: str = ""
    tp_price: float = 0.0
    exit_reason: str = ""


class State:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, isolation_level=None)  # autocommit
        self.db.executescript(SCHEMA)

    # --- key/value ---------------------------------------------------------------------
    def get(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key: str, value) -> None:
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, json.dumps(value)))

    # --- plans ---------------------------------------------------------------------------
    def plan(self, symbol: str) -> Plan | None:
        cur = self.db.execute("SELECT * FROM plans WHERE symbol = ?", (symbol,))
        row = cur.fetchone()
        if row is None:
            return None
        names = [d[0] for d in cur.description]
        return Plan(**dict(zip(names, row, strict=True)))

    def save_plan(self, plan: Plan) -> None:
        names = [f.name for f in fields(Plan)]
        self.db.execute(
            f"INSERT OR REPLACE INTO plans ({', '.join(names)}) "
            f"VALUES ({', '.join('?' * len(names))})",
            [asdict(plan)[n] for n in names])

    def delete_plan(self, symbol: str) -> None:
        self.db.execute("DELETE FROM plans WHERE symbol = ?", (symbol,))

    # --- journal -------------------------------------------------------------------------
    def add_trade(self, **t) -> None:
        cols = ", ".join(t)
        self.db.execute(f"INSERT INTO trades ({cols}) VALUES ({', '.join('?' * len(t))})",
                        list(t.values()))

    def trades(self) -> list[dict]:
        cur = self.db.execute("SELECT * FROM trades ORDER BY id")
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]

    def add_signal(self, bar: str, symbol: str, probs: tuple[float, float, float], action: str,
                   detail: str = "") -> None:
        self.db.execute("INSERT OR REPLACE INTO signals VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (bar, symbol, *map(float, probs), action, detail))

    def last_signals(self, n: int) -> list[dict]:
        cur = self.db.execute("SELECT * FROM signals ORDER BY bar DESC, symbol LIMIT ?", (n,))
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]
