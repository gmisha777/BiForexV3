"""Compute the feature table for each configured symbol and save it to data/features/.

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --symbols ETHUSDT
"""

from __future__ import annotations

import argparse
import logging
import sys

from biforex.config import load_settings
from biforex.data import store
from biforex.features.pipeline import build_features, features_path
from biforex.log import setup_logging

log = logging.getLogger("build_features")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("--symbols", nargs="+", help="override symbols from config")
    args = parser.parse_args()

    settings = load_settings(args.config)
    setup_logging(settings.logging.level, settings.logging.dir)

    for symbol in args.symbols or settings.all_symbols:
        features = build_features(settings, symbol)
        path = features_path(settings, symbol)
        store.save(features.reset_index(), path)

        nan_share = features.isna().mean()
        log.info("%s: %d rows x %d features, %s -> %s, saved to %s", symbol, len(features),
                 features.shape[1], features.index[0], features.index[-1], path)
        partial = nan_share[nan_share > 0.001].sort_values(ascending=False)
        for group, share in partial.groupby(partial.index.str.split("_").str[0]).max().items():
            log.info("    %-10s up to %.1f%% missing", group, 100 * share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
