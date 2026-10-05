"""Strategy lgbm_tb: LightGBM on the full feature set (the original BiForexV3 model).

Features: src/biforex/features (trend, oscillators, volatility, volume, funding, open
interest, BTC context, higher timeframes). Settings: the `model:` section of config.yaml;
keys in strategies.lgbm_tb.params override it. Scoring: see lgbm_base.
"""

from __future__ import annotations

import pandas as pd

from ..config import Settings
from ..data import store
from ..features.pipeline import build_features, features_path
from . import register
from .lgbm_base import LgbmStrategy, probs_to_score  # noqa: F401  (re-exported)


@register
class LgbmTripleBarrier(LgbmStrategy):
    name = "lgbm_tb"

    def base_model_config(self, settings: Settings) -> dict:
        return settings.model.model_dump()

    def features(self, symbol: str, live: bool = False) -> pd.DataFrame:
        if not live:  # stored by scripts/build_features.py
            stored = store.load(features_path(self.st, symbol))
            if stored is not None:
                return stored.set_index("open_time")
        return build_features(self.st, symbol)
