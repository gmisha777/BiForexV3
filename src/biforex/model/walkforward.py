"""Walk-forward training: monthly test windows, rolling training window, purged boundaries.

Labels look `horizon` bars ahead, so a training row close to the test start would contain
information from the test period. Such rows are purged; the same applies to the boundary
between the fitting part and the early-stopping part of the training window.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import Settings
from ..data.schema import timeframe_delta
from . import CLASSES, PROBA_COLUMNS
from .dataset import Dataset

log = logging.getLogger(__name__)


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # exclusive, already purged
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # exclusive


def monthly_folds(
    index: pd.DatetimeIndex, train_months: int, purge: pd.Timedelta
) -> Iterator[Fold]:
    first_month = index[0].replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    test_start = first_month + pd.DateOffset(months=train_months)
    while test_start <= index[-1]:
        test_end = test_start + pd.DateOffset(months=1)
        yield Fold(test_start - pd.DateOffset(months=train_months), test_start - purge,
                   test_start, test_end)
        test_start = test_end


def fit(
    X: pd.DataFrame, y: pd.Series, settings: Settings, purge: pd.Timedelta
) -> lgb.LGBMClassifier:
    """Fit on a training window; its tail (after a purge gap) drives early stopping."""
    mc = settings.model
    val_start = X.index[int(len(X) * (1 - mc.val_fraction))]
    fit_mask = X.index < val_start - purge
    val_mask = X.index >= val_start
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=len(CLASSES), verbose=-1, n_jobs=-1,
        random_state=42, **mc.params,
    )
    model.fit(
        X[fit_mask], y[fit_mask],
        eval_X=X[val_mask], eval_y=y[val_mask],
        callbacks=[lgb.early_stopping(mc.early_stopping_rounds, verbose=False)],
    )
    return model


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame  # out-of-sample p_down/p_flat/p_up + outcomes + fold id
    importance: pd.DataFrame  # gain per feature per fold
    folds: pd.DataFrame  # fold boundaries, sizes, trees, train class priors


def run_walkforward(ds: Dataset, settings: Settings) -> WalkForwardResult:
    purge = (settings.label.horizon_bars + 1) * timeframe_delta(settings.timeframe)
    preds, importance, fold_rows = [], {}, []
    for k, fold in enumerate(monthly_folds(ds.X.index, settings.retrain.train_months, purge)):
        tr = (ds.X.index >= fold.train_start) & (ds.X.index < fold.train_end)
        te = (ds.X.index >= fold.test_start) & (ds.X.index < fold.test_end)
        if te.sum() == 0:
            continue
        model = fit(ds.X[tr], ds.y[tr], settings, purge)
        p = ds.outcomes[te].copy()  # positional: safe when timestamps repeat (pooled symbols)
        p[list(PROBA_COLUMNS)] = model.predict_proba(ds.X[te])
        p["fold"] = k
        preds.append(p)

        importance[fold.test_start] = pd.Series(
            model.booster_.feature_importance("gain"), index=ds.X.columns)
        prior = ds.y[tr].value_counts(normalize=True).reindex(range(len(CLASSES)), fill_value=0)
        fold_rows.append({
            "fold": k, "test_start": fold.test_start, "train_rows": int(tr.sum()),
            "test_rows": int(te.sum()), "trees": model.best_iteration_,
            **{f"prior_{c}": prior.iloc[i] for i, c in enumerate(PROBA_COLUMNS)},
        })
        log.info("fold %2d test %s: train %d, test %d, trees %d", k,
                 fold.test_start.date(), tr.sum(), te.sum(), model.best_iteration_)

    return WalkForwardResult(
        pd.concat(preds), pd.DataFrame(importance).T, pd.DataFrame(fold_rows).set_index("fold"))


def fit_final(ds: Dataset, settings: Settings) -> lgb.LGBMClassifier:
    """Model for live use: trained on the most recent training window."""
    purge = (settings.label.horizon_bars + 1) * timeframe_delta(settings.timeframe)
    start = ds.X.index[-1] - pd.DateOffset(months=settings.retrain.train_months)
    mask = np.asarray(ds.X.index > start)
    return fit(ds.X[mask], ds.y[mask], settings, purge)
