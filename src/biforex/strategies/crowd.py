"""Strategy crowd: LightGBM on crowd positioning only.

Futures positioning (funding, open interest, top-trader and global long/short ratios, taker
buy/sell ratio) and how extreme it is versus the last month, plus a minimal price context.
In lgbm_tb these features rank high but are mixed with ~70 others; a small, strongly
regularized model on them alone gives a more independent opinion for the ensemble.
Open interest / long-short data start in 2020-09 (BTC) and 2021-12 (ETH); earlier rows are
NaN, which LightGBM handles.
"""

from __future__ import annotations

import pandas as pd

from . import register
from .lgbm_tb import LgbmTripleBarrier

POSITIONING = [
    "funding_last", "funding_mean3", "funding_sum21",
    "oi_chg_1", "oi_chg_4", "oi_chg_24", "oi_z168", "oi_to_volume24",
    "top_ls_accounts_log", "top_ls_accounts_chg24", "top_ls_positions_log",
    "top_ls_positions_chg24", "global_ls_log", "global_ls_chg24",
    "taker_ls_1h_log", "taker_ls_1h_chg24", "taker_imbalance24",
]
CONTEXT = ["ret_1", "ret_8", "ret_34", "rv24", "atr_pct"]
Z_WINDOW = 180  # 30 days of 4h bars
Z_COLUMNS = ["funding_mean3", "top_ls_positions_log", "global_ls_log", "taker_ls_1h_log"]


@register
class Crowd(LgbmTripleBarrier):
    name = "crowd"
    model_defaults = {"params": {
        "n_estimators": 3000, "learning_rate": 0.01, "num_leaves": 4,
        "min_child_samples": 500, "subsample": 0.7, "subsample_freq": 1,
        "colsample_bytree": 0.8, "reg_lambda": 20.0,
    }}

    def base_model_config(self, settings) -> dict:
        return {**settings.model.model_dump(), **self.model_defaults}

    def features(self, symbol: str, live: bool = False) -> pd.DataFrame:
        full = super().features(symbol, live)
        f = full[[c for c in POSITIONING + CONTEXT if c in full.columns]].copy()
        for c in Z_COLUMNS:
            if c in f:
                x = f[c].astype("float64")
                mean = x.rolling(Z_WINDOW, min_periods=Z_WINDOW // 2).mean()
                std = x.rolling(Z_WINDOW, min_periods=Z_WINDOW // 2).std()
                f[f"{c}_z"] = ((x - mean) / std).astype("float32")
        return f
