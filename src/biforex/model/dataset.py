"""Join stored features and labels into a training table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Settings
from ..data import store
from ..features.pipeline import features_path
from ..labels.build import labels_path
from . import CLASSES

OUTCOME_COLUMNS = ["label", "long_outcome", "long_ret", "long_bars",
                   "short_outcome", "short_ret", "short_bars", "atr"]


@dataclass
class Dataset:
    X: pd.DataFrame  # features, indexed by open_time
    y: pd.Series  # class index 0..2 (see CLASSES)
    outcomes: pd.DataFrame  # label, per-side returns etc. for trading metrics

    def __len__(self) -> int:
        return len(self.y)


def load_dataset(settings: Settings, symbol: str) -> Dataset:
    features = store.load(features_path(settings, symbol))
    labels = store.load(labels_path(settings, symbol))
    if features is None or labels is None:
        raise FileNotFoundError(
            f"{symbol}: run scripts/build_features.py and scripts/build_labels.py first")
    return make_dataset(features.set_index("open_time"), labels.set_index("open_time"))


def make_dataset(features: pd.DataFrame, labels: pd.DataFrame) -> Dataset:
    """features/labels indexed by open_time -> aligned dataset of labelled rows."""
    labels = labels.dropna(subset=["label"])
    idx = features.index.intersection(labels.index)
    outcomes = labels.loc[idx, OUTCOME_COLUMNS]
    y = outcomes["label"].map({c: i for i, c in enumerate(CLASSES)}).astype("int8")
    return Dataset(features.loc[idx], y, outcomes)


def pool(datasets: dict[str, Dataset]) -> Dataset:
    """One dataset over several symbols: rows sorted by time, `symbol_id` feature added,
    outcomes get a `symbol` column."""
    Xs, ys, outs = [], [], []
    for i, (symbol, ds) in enumerate(datasets.items()):
        Xs.append(ds.X.assign(symbol_id=np.float32(i)))
        ys.append(ds.y)
        outs.append(ds.outcomes.assign(symbol=symbol))
    X = pd.concat(Xs)
    order = X.index.argsort(kind="stable")
    return Dataset(X.iloc[order], pd.concat(ys).iloc[order], pd.concat(outs).iloc[order])
