"""Walk-forward validation + final model for each strategy.

Usage:
    python scripts/train.py                          # all enabled strategies in config
    python scripts/train.py --strategies lgbm_tb     # chosen strategies
    python scripts/train.py --no-final               # validation only, keep live models
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from biforex.config import load_settings
from biforex.log import setup_logging
from biforex.model import evaluate as ev
from biforex.strategies import REGISTRY, create, save_oos_scores

log = logging.getLogger("train")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("--strategies", nargs="+",
                        help=f"strategies to train (available: {', '.join(sorted(REGISTRY))})")
    parser.add_argument("--no-final", action="store_true", help="skip training the final model")
    args = parser.parse_args()

    settings = load_settings(args.config)
    setup_logging(settings.logging.level, settings.logging.dir)
    logging.getLogger("biforex.model.walkforward").setLevel(logging.WARNING)
    cost = ev.round_trip_cost(settings.execution.fee_taker, settings.execution.slippage_bps)
    names = args.strategies or list(settings.enabled_strategies)

    for name in names:
        strategy = create(name, settings)
        log.info("=== %s: walk-forward on %s bars (train %d months, test 1 month)...", name,
                 settings.timeframe, settings.retrain.train_months)
        wf = strategy.walk_forward()
        save_oos_scores(strategy, wf.scores)
        for rname, df in wf.reports.items():
            df.to_csv(strategy.report_dir / f"{rname}.csv")
        if wf.summary:
            log.info("%s:\n%s", name, wf.summary)

        traded = wf.scores[wf.scores["symbol"].isin(settings.symbols)]
        scored = ev.attach_outcomes(traded, settings)
        log.info("%s scores out-of-sample %s -> %s, distribution: %s", name,
                 scored.index[0].date(), scored.index[-1].date(),
                 scored["score"].describe(percentiles=[.05, .5, .95]).round(3).to_dict())
        for symbol, g in scored.groupby("symbol"):
            table = ev.score_signal_table(g, cost)
            table.to_csv(strategy.report_dir / f"signals_{symbol}.csv")
            log.info("%s %s signals (every bar a separate trade, net of %.2f%% costs):\n%s",
                     name, symbol, 100 * cost, table[table["signals"] > 0].round(3).to_string())

        if not args.no_final:
            strategy.fit_final()
    return 0


if __name__ == "__main__":
    sys.exit(main())
