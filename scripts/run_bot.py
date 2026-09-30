"""Run the trading bot.

    python scripts/run_bot.py                  # demo account (execution.testnet: true)
    python scripts/run_bot.py --dry-run --once # one decision cycle, no orders
    python scripts/run_bot.py --live           # REAL account: needs execution.testnet: false

Create a file named STOP in the project root to block new entries (open positions keep their
stop and take profit). Delete data/bot/<mode>.sqlite key "halted" to resume after the
max-drawdown halt (see scripts/bot_status.py --reset-halt).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

from biforex import notify
from biforex.config import load_settings
from biforex.log import setup_logging
from biforex.trading.binance import BinanceFutures
from biforex.trading.bot import Bot
from biforex.trading.live_data import Predictor
from biforex.trading.state import State

log = logging.getLogger("run_bot")


def keep_awake() -> None:
    """Stop Windows from sleeping while the bot runs (the display may still turn off)."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def single_instance(path) -> object:
    """Exclusive lock so two bots never trade the same account."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+")  # noqa: SIM115 - kept open for the process lifetime
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"another bot instance holds {path}")
    return f


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("--live", action="store_true", help="trade the REAL account")
    parser.add_argument("--dry-run", action="store_true", help="decide and log, place no orders")
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    parser.add_argument("--force", metavar="SYMBOL:long|short",
                        help="demo only: force a signal on the next bar close (order test)")
    args = parser.parse_args()

    settings = load_settings(args.config)
    demo = settings.execution.testnet
    if args.live and demo:
        sys.exit("--live given but execution.testnet is true in config; refusing")
    if not args.live and not demo:
        sys.exit("execution.testnet is false (real account): add --live to confirm")
    mode = "demo" if demo else "live"

    setup_logging(settings.logging.level, settings.logging.dir)
    lock = single_instance(settings.data.dir / "bot" / f"{mode}.lock")  # noqa: F841
    prefix = "BINANCE_TESTNET_API" if demo else "BINANCE_API"
    client = BinanceFutures(os.environ.get(f"{prefix}_KEY", ""),
                            os.environ.get(f"{prefix}_SECRET", ""), demo=demo)
    state = State(settings.data.dir / "bot" / f"{mode}{'_dry' if args.dry_run else ''}.sqlite")
    notifier = notify.from_env() if not (args.dry_run or args.once) else notify.Notifier()
    bot = Bot(settings, client, state, Predictor(settings), dry_run=args.dry_run,
              notifier=notifier)
    log.info("telegram notifications: %s", "on" if notifier.enabled else "off")
    equity, available = client.usdt_equity()
    log.info("%s account: equity %.4f USDT, available %.4f; symbols %s", mode.upper(), equity,
             available, settings.symbols)
    bot.setup()
    if args.force:
        if not demo:
            sys.exit("--force is allowed on the demo account only")
        sym, side = args.force.split(":")
        bot.forced[sym] = 1 if side == "long" else -1
        bot.state.set("last_bar", None)  # process the latest closed bar right away

    if args.once:
        bar = bot.last_closed_bar(pd.Timestamp.now(tz="UTC"))
        for s in settings.symbols:
            bot.sync(s)
        bot.on_bar_close(bar)
        return 0
    keep_awake()
    bot.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
