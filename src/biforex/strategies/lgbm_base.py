"""Common machinery for strategies that are LightGBM classifiers on the shared labels.

A subclass only says which features it uses (`features`); walk-forward validation, the final
model, loading and scoring are the same for all of them. The model predicts
p_down / p_flat / p_up for "which barrier is hit first within the horizon" and the score is
the signed strength of the dominant direction above the neutral 1/3 level:
    score = +(p_up - 1/3) / (2/3)    if p_up > p_down   (0 if p_up < 1/3)
    score = -(p_down - 1/3) / (2/3)  otherwise
so |score| >= 0.10 means "p >= 0.40 and dominant". (p_up - p_down was tried for lgbm_tb and
selected worse bars in the backtest.)
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np
import pandas as pd

from ..config import ModelConfig, Settings
from ..data import store
from ..labels.build import labels_path
from ..model import PROBA_COLUMNS
from ..model import evaluate as ev
from ..model.dataset import Dataset, make_dataset, pool
from ..model.registry import load_current, save_model
from ..model.walkforward import fit_final, run_walkforward
from .base import Strategy, WalkForwardScores, clip_score

log = logging.getLogger(__name__)

NEUTRAL = 1 / 3


def probs_to_score(p_down, p_up):
    """Vectorized: dominant direction's probability above 1/3, scaled to 0..1, signed."""
    p_down, p_up = np.asarray(p_down, dtype=float), np.asarray(p_up, dtype=float)
    up = p_up > p_down
    strength = (np.where(up, p_up, p_down) - NEUTRAL) / (1 - NEUTRAL)
    return np.where(up, 1, -1) * np.clip(strength, 0, 1)


class LgbmStrategy(Strategy):
    """Subclasses set `name` and implement `features`; optionally `model_defaults`."""

    model_defaults: ClassVar[dict] = {}  # overrides of ModelConfig for this strategy

    def __init__(self, settings: Settings, params: dict | None = None):
        super().__init__(settings, params)
        model_cfg = ModelConfig(**{**self.base_model_config(settings), **self.model_defaults,
                                   **self.params})
        self.st = settings.model_copy(update={"model": model_cfg})
        self.models: dict[str, tuple] = {}  # model name -> (booster, meta)

    def base_model_config(self, settings: Settings) -> dict:
        return ModelConfig().model_dump()

    # --- to implement ------------------------------------------------------------------
    def features(self, symbol: str, live: bool = False) -> pd.DataFrame:
        """Feature table indexed by open_time, one row per closed bar (float32).
        live=True: called by the bot, must include the latest closed bar."""
        raise NotImplementedError

    # --- research ------------------------------------------------------------------------
    def dataset(self, symbol: str) -> Dataset:
        labels = store.load(labels_path(self.st, symbol))
        if labels is None:
            raise FileNotFoundError(f"{symbol}: run scripts/build_labels.py first")
        return make_dataset(self.features(symbol), labels.set_index("open_time"))

    def _jobs(self) -> dict[str, tuple[Dataset, list[str]]]:
        """Model name -> (dataset, symbols in symbol_id order)."""
        st = self.st
        if st.model.pooled:
            symbols = [s for s in st.all_symbols if self._has_data(s)]
            return {self.name: (pool({s: self.dataset(s) for s in symbols}), symbols)}
        return {f"{self.name}/{s}": (self.dataset(s), [s]) for s in st.symbols}

    def _has_data(self, symbol: str) -> bool:
        return True

    def walk_forward(self) -> WalkForwardScores:
        preds, reports, lines = [], {}, []
        for name, (ds, symbols) in self._jobs().items():
            wf = run_walkforward(ds, self.st)
            p = wf.predictions
            if "symbol" not in p:
                p = p.assign(symbol=symbols[0])
            preds.append(p)
            tag = name.replace("/", "_")
            quality = ev.probability_quality(p, wf.folds)
            reports[f"quality_{tag}"] = quality
            reports[f"top_features_{tag}"] = ev.top_features(wf.importance)
            reports[f"folds_{tag}"] = wf.folds
            lines.append(f"{name}: {len(wf.folds)} folds, median {wf.folds['trees'].median():.0f} "
                         f"trees; log-loss vs class frequencies:\n"
                         f"{quality.round(3).to_string()}\ntop features:\n"
                         f"{ev.top_features(wf.importance, 12).round(2).to_string()}")
        p = pd.concat(preds).sort_index(kind="stable")
        p["score"] = probs_to_score(p["p_down"].to_numpy(), p["p_up"].to_numpy())
        self.report_dir.mkdir(parents=True, exist_ok=True)
        p.to_parquet(self.report_dir / "oos_predictions.parquet")
        return WalkForwardScores(p[["symbol", "score"]], reports, "\n".join(lines))

    def fit_final(self) -> dict:
        metas = {}
        for name, (ds, symbols) in self._jobs().items():
            model = fit_final(ds, self.st)
            meta = {
                "trained_until": ds.X.index[-1], "timeframe": self.st.timeframe,
                "symbols": symbols,  # order = symbol_id
                "features": list(ds.X.columns), "trees": model.best_iteration_,
                "label": self.st.label.model_dump(), "model": self.st.model.model_dump(),
            }
            folder = save_model(name, model, meta, make_current=True)
            log.info("%s: final model (%d trees) saved to %s", name, model.best_iteration_,
                     folder)
            metas[name] = meta
        return metas

    # --- live ------------------------------------------------------------------------------
    def load(self) -> None:
        names = ([self.name] if self.st.model.pooled
                 else [f"{self.name}/{s}" for s in self.st.symbols])
        for name in names:
            booster, meta = load_current(name)
            if meta["timeframe"] != self.st.timeframe:
                raise RuntimeError(f"{name}: model timeframe {meta['timeframe']} != config "
                                   f"{self.st.timeframe}; retrain")
            self.models[name] = (booster, meta)
            log.info("%s/%s: trained until %s, %d trees, symbols %s", name, meta["version"],
                     meta["trained_until"], meta["trees"], meta["symbols"])

    def probabilities(self, symbol: str, bar: pd.Timestamp) -> np.ndarray | None:
        name = self.name if self.st.model.pooled else f"{self.name}/{symbol}"
        booster, meta = self.models[name]
        features = self.features(symbol, live=True)
        if bar not in features.index:
            log.warning("%s %s: no features for bar %s (last %s)", self.name, symbol, bar,
                        features.index[-1] if len(features) else None)
            return None
        row = features.loc[[bar]].copy()
        if "symbol_id" in meta["features"]:
            row["symbol_id"] = np.float32(meta["symbols"].index(symbol))
        return booster.predict(row.reindex(columns=meta["features"]))[0]

    def score(self, symbol: str, bar: pd.Timestamp) -> float | None:
        p = self.probabilities(symbol, bar)
        if p is None:
            return None
        score = clip_score(float(probs_to_score(p[0], p[2])))
        log.info("%s %s bar %s | %s | score %+.3f", self.name, symbol, bar,
                 " ".join(f"{c}={v:.3f}" for c, v in zip(PROBA_COLUMNS, p, strict=True)), score)
        return score
