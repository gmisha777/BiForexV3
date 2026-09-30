"""Walk-forward validation + final model (pooled over all symbols, or one per symbol).

Usage:
    python scripts/train.py                 # validate, report, save final model
    python scripts/train.py --no-final      # validation report only
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from biforex.config import PROJECT_ROOT, load_settings
from biforex.log import setup_logging
from biforex.model import evaluate as ev
from biforex.model.dataset import load_dataset, pool
from biforex.model.registry import POOLED, save_model
from biforex.model.walkforward import fit_final, run_walkforward

log = logging.getLogger("train")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def fmt(df: pd.DataFrame) -> str:
    return df.round(3).to_string()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("--symbols", nargs="+", help="override symbols from config")
    parser.add_argument("--no-final", action="store_true", help="skip training the final model")
    args = parser.parse_args()

    settings = load_settings(args.config)
    setup_logging(settings.logging.level, settings.logging.dir)
    logging.getLogger("biforex.model.walkforward").setLevel(logging.WARNING)
    cost = ev.round_trip_cost(settings.execution.fee_taker, settings.execution.slippage_bps)
    symbols = args.symbols or (settings.all_symbols if settings.model.pooled
                                else settings.symbols)

    datasets = {s: load_dataset(settings, s) for s in symbols}
    if settings.model.pooled:
        jobs = {POOLED: pool(datasets)}
    else:
        jobs = datasets

    for name, ds in jobs.items():
        log.info("%s: %d rows, %d features, timeframe %s; walk-forward (train %d months, "
                 "test 1 month)...", name, len(ds), ds.X.shape[1], settings.timeframe,
                 settings.retrain.train_months)
        wf = run_walkforward(ds, settings)
        preds = wf.predictions
        if "symbol" not in preds:
            preds = preds.assign(symbol=name)

        out = PROJECT_ROOT / "reports" / name
        out.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(out / "oos_predictions.parquet")
        quality = ev.probability_quality(preds, wf.folds)
        top = ev.top_features(wf.importance)
        for fname, df in [("quality", quality), ("top_features", top), ("folds", wf.folds)]:
            df.to_csv(out / f"{fname}.csv")

        log.info("%s out-of-sample %s -> %s, %d folds, median %d trees, cost %.2f%% per trade",
                 name, preds.index[0].date(), preds.index[-1].date(), len(wf.folds),
                 wf.folds["trees"].median(), 100 * cost)
        log.info("probability quality (log-loss vs class frequencies):\n%s", fmt(quality))
        for symbol, g in preds.groupby("symbol"):
            signals = ev.signal_table(g, cost)
            signals.to_csv(out / f"signals_{symbol}.csv")
            log.info("%s signals (every bar = separate trade, net of costs):\n%s", symbol,
                     fmt(signals[signals["signals"] > 0]))
        log.info("top features:\n%s", fmt(top))

        if not args.no_final:
            model = fit_final(ds, settings)
            folder = save_model(name, model, {
                "trained_until": ds.X.index[-1],
                "timeframe": settings.timeframe,
                "symbols": list(datasets) if name == POOLED else [name],  # order = symbol_id
                "features": list(ds.X.columns),
                "trees": model.best_iteration_,
                "label": settings.label.model_dump(),
                "model": settings.model.model_dump(),
                "oos_logloss_improvement_pct": float(quality.loc["total", "improvement_pct"]),
            }, make_current=True)
            log.info("%s final model (%d trees) saved to %s", name, model.best_iteration_,
                     folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
