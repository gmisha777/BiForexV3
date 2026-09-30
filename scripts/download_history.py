"""Download or incrementally update market history for the configured symbols.

Usage:
    python scripts/download_history.py
    python scripts/download_history.py --symbols BTCUSDT --intervals 1h --no-metrics
"""

from __future__ import annotations

import argparse
import logging
import sys

from biforex.config import load_settings
from biforex.data.update import make_session, update_funding, update_klines, update_metrics
from biforex.log import setup_logging

log = logging.getLogger("download_history")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("--symbols", nargs="+", help="override symbols from config")
    parser.add_argument("--intervals", nargs="+", help="override kline intervals from config")
    parser.add_argument("--no-funding", action="store_true")
    parser.add_argument("--no-metrics", action="store_true")
    args = parser.parse_args()

    settings = load_settings(args.config)
    setup_logging(settings.logging.level, settings.logging.dir)
    symbols = args.symbols or settings.all_symbols
    intervals = args.intervals or settings.data.intervals
    session = make_session(settings.data.download_workers)

    all_ok = True
    for symbol in symbols:
        for interval in intervals:
            report = update_klines(settings, symbol, interval, session)
            log.info("%s %s: %s", symbol, interval, report.summary())
            all_ok &= report.ok
        if settings.data.funding and not args.no_funding:
            update_funding(settings, symbol, session)
        if settings.data.metrics and not args.no_metrics:
            update_metrics(settings, symbol, session)

    log.info("done, data in %s", settings.data.dir)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
