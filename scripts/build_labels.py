"""Compute triple-barrier labels for each configured symbol and save them to data/labels/.

Usage:
    python scripts/build_labels.py
    python scripts/build_labels.py --symbols BTCUSDT
"""

from __future__ import annotations

import argparse
import logging
import sys

from biforex.config import load_settings
from biforex.data import store
from biforex.labels.build import build_labels, labels_path
from biforex.log import setup_logging

log = logging.getLogger("build_labels")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("--symbols", nargs="+", help="override symbols from config")
    args = parser.parse_args()

    settings = load_settings(args.config)
    setup_logging(settings.logging.level, settings.logging.dir)
    lc = settings.label
    log.info("barriers: TP %.1f ATR, SL %.1f ATR, horizon %d bars", lc.tp_atr, lc.sl_atr,
             lc.horizon_bars)

    for symbol in args.symbols or settings.all_symbols:
        labels = build_labels(settings, symbol)
        store.save(labels.reset_index(), labels_path(settings, symbol))
        done = labels.dropna(subset=["label"])

        share = done["label"].value_counts(normalize=True).reindex([1, 0, -1]).fillna(0)
        log.info("%s: %d labelled bars | up %.1f%%, flat %.1f%%, down %.1f%%", symbol, len(done),
                 *(100 * share.to_numpy()))
        by_year = done.groupby(done.index.year)["label"].value_counts(normalize=True).unstack()
        log.info("share by year (1=up, 0=flat, -1=down):\n%s", (100 * by_year).round(1))
        for side in ("long", "short"):
            win = (done[f"{side}_outcome"] == 1).mean()
            log.info("  always %-5s: TP first %.1f%%, mean return %+.3f%% per trade, "
                     "avg holding %.1f bars (before fees)", side, 100 * win,
                     100 * done[f"{side}_ret"].mean(), done[f"{side}_bars"].mean())
    return 0


if __name__ == "__main__":
    sys.exit(main())
