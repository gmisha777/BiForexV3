"""Strategy som_regime: market regimes by a self-organizing map, one small model per regime.

A modern take on "block 4" of the legacy MATLAB system: a Kohonen map (SOM) groups market
states into regimes using a dozen regime features (volatility, trend, volume, positioning);
each regime with enough training rows gets its own small LightGBM on the full lgbm_tb feature
set, rare regimes fall back to their class frequencies. Everything (scaling, SOM, models) is
fitted inside each walk-forward training window. Score: as in lgbm_base.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from minisom import MiniSom

from ..data.schema import timeframe_delta
from ..model import CLASSES, PROBA_COLUMNS
from ..model import evaluate as ev
from ..model.registry import MODELS_DIR
from ..model.walkforward import monthly_folds
from . import register
from .base import Strategy, WalkForwardScores, clip_score
from .lgbm_base import probs_to_score
from .lgbm_tb import LgbmTripleBarrier

log = logging.getLogger(__name__)

DEFAULTS = {
    "grid": 4,  # 4 x 4 map = 16 regimes, as in the legacy system
    "min_rows": 400,  # a regime needs this many training rows for its own model
    "som_iterations": 10000,
    "regime_features": [
        "atr_pct", "rv_ratio", "bb_width", "vol_rel24", "ema55_dist_atr", "ema144_slope_atr",
        "h1d_ret_3", "h1d_rsi14", "dist_high168_atr", "oi_z168", "funding_sum21",
        "global_ls_log",
    ],
    "lgbm": {"n_estimators": 150, "learning_rate": 0.03, "num_leaves": 4,
             "min_child_samples": 100, "subsample": 0.7, "subsample_freq": 1,
             "colsample_bytree": 0.7, "reg_lambda": 10.0},
}


class RegimeModel:
    def __init__(self, cfg: dict, seed: int = 42):
        self.cfg, self.seed = cfg, seed

    def _z(self, X: pd.DataFrame) -> np.ndarray:
        R = X[self.cols].astype("float64")
        return ((R - self.med) / self.scale).fillna(0).clip(-5, 5).to_numpy()

    def clusters(self, X: pd.DataFrame) -> np.ndarray:
        Z = self._z(X)
        W = self.som.get_weights().reshape(-1, Z.shape[1])
        d2 = (Z**2).sum(1)[:, None] - 2 * Z @ W.T + (W**2).sum(1)[None, :]
        return d2.argmin(1)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RegimeModel:
        self.cols = [c for c in self.cfg["regime_features"] if c in X.columns]
        R = X[self.cols].astype("float64")
        self.med = R.median()
        self.scale = (R.quantile(0.75) - R.quantile(0.25)).replace(0, 1).fillna(1)
        Z = self._z(X)
        g = self.cfg["grid"]
        self.som = MiniSom(g, g, Z.shape[1], sigma=1.0, learning_rate=0.5,
                           random_seed=self.seed)
        self.som.pca_weights_init(Z)
        self.som.train_random(Z, self.cfg["som_iterations"])
        cl = self.clusters(X)
        yv = y.to_numpy()
        self.prior = np.bincount(yv, minlength=len(CLASSES)) / len(yv)
        self.models, self.priors, self.sizes = {}, {}, {}
        for c in np.unique(cl):
            rows = cl == c
            n = int(rows.sum())
            self.sizes[int(c)] = n
            counts = np.bincount(yv[rows], minlength=len(CLASSES))
            self.priors[int(c)] = (counts + 100 * self.prior) / (n + 100)  # shrunk to overall
            if n >= self.cfg["min_rows"] and (counts > 0).all():
                m = lgb.LGBMClassifier(objective="multiclass", num_class=len(CLASSES),
                                       verbose=-1, n_jobs=-1, random_state=self.seed,
                                       **self.cfg["lgbm"])
                m.fit(X[rows], yv[rows])
                self.models[int(c)] = m
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cl = self.clusters(X)
        out = np.tile(self.prior, (len(X), 1))
        for c in np.unique(cl):
            rows = cl == c
            if int(c) in self.models:
                out[rows] = self.models[int(c)].predict_proba(X[rows])
            elif int(c) in self.priors:
                out[rows] = self.priors[int(c)]
        return out


@register
class SomRegime(Strategy):
    name = "som_regime"

    def __init__(self, settings, params: dict | None = None):
        super().__init__(settings, params)
        self.cfg = {**DEFAULTS, **self.params}
        self.features_src = LgbmTripleBarrier(settings)
        self.model: RegimeModel | None = None
        self.meta: dict = {}

    def _dataset(self):
        (ds, symbols), = self.features_src._jobs().values()
        return ds, symbols

    def walk_forward(self) -> WalkForwardScores:
        st = self.settings
        ds, _ = self._dataset()
        purge = (st.label.horizon_bars + 1) * timeframe_delta(st.timeframe)
        preds, fold_rows = [], []
        for k, fold in enumerate(monthly_folds(ds.X.index, st.retrain.train_months, purge)):
            tr = (ds.X.index >= fold.train_start) & (ds.X.index < fold.train_end)
            te = (ds.X.index >= fold.test_start) & (ds.X.index < fold.test_end)
            if te.sum() == 0:
                continue
            m = RegimeModel(self.cfg).fit(ds.X[tr], ds.y[tr])
            p = ds.outcomes[te].copy()
            p[list(PROBA_COLUMNS)] = m.predict_proba(ds.X[te])
            p["fold"] = k
            preds.append(p)
            prior = ds.y[tr].value_counts(normalize=True).reindex(range(3), fill_value=0)
            fold_rows.append({"fold": k, "test_start": fold.test_start,
                              "regimes_with_model": len(m.models), "regimes": len(m.sizes),
                              **{f"prior_{c}": prior.iloc[i]
                                 for i, c in enumerate(PROBA_COLUMNS)}})
        p = pd.concat(preds).sort_index(kind="stable")
        folds = pd.DataFrame(fold_rows).set_index("fold")
        p["score"] = probs_to_score(p["p_down"].to_numpy(), p["p_up"].to_numpy())
        self.report_dir.mkdir(parents=True, exist_ok=True)
        p.to_parquet(self.report_dir / "oos_predictions.parquet")
        quality = ev.probability_quality(p, folds)
        summary = (f"{len(folds)} folds; regimes with own model per fold: median "
                   f"{folds['regimes_with_model'].median():.0f} of {folds['regimes'].median():.0f}"
                   f"\nlog-loss vs class frequencies:\n{quality.round(3).to_string()}")
        return WalkForwardScores(p[["symbol", "score"]], {"quality": quality, "folds": folds},
                                 summary)

    def fit_final(self) -> dict:
        st = self.settings
        ds, symbols = self._dataset()
        start = ds.X.index[-1] - pd.DateOffset(months=st.retrain.train_months)
        mask = np.asarray(ds.X.index > start)
        model = RegimeModel(self.cfg).fit(ds.X[mask], ds.y[mask])
        version = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        folder = MODELS_DIR / self.name / version
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, folder / "model.pkl")
        meta = {"name": self.name, "version": version, "timeframe": st.timeframe,
                "trained_until": str(ds.X.index[-1]), "symbols": symbols,
                "features": list(ds.X.columns), "regimes_with_model": len(model.models)}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (MODELS_DIR / self.name / "current.json").write_text(json.dumps({"version": version}),
                                                            encoding="utf-8")
        log.info("%s: saved to %s (%d regimes with own model)", self.name, folder,
                 len(model.models))
        return meta

    def load(self) -> None:
        version = json.loads((MODELS_DIR / self.name / "current.json").read_text())["version"]
        folder = MODELS_DIR / self.name / version
        self.meta = json.loads((folder / "meta.json").read_text())
        if self.meta["timeframe"] != self.settings.timeframe:
            raise RuntimeError(f"{self.name}: trained on {self.meta['timeframe']}; retrain")
        self.model = joblib.load(folder / "model.pkl")
        log.info("%s/%s: trained until %s", self.name, version, self.meta["trained_until"])

    def score(self, symbol: str, bar: pd.Timestamp) -> float | None:
        features = self.features_src.features(symbol, live=True)
        if bar not in features.index:
            return None
        row = features.loc[[bar]].copy()
        if "symbol_id" in self.meta["features"]:
            row["symbol_id"] = np.float32(self.meta["symbols"].index(symbol))
        p = self.model.predict_proba(row.reindex(columns=self.meta["features"]))[0]
        score = clip_score(float(probs_to_score(p[0], p[2])))
        log.info("%s %s bar %s | regime %d | score %+.3f", self.name, symbol, bar,
                 int(self.model.clusters(row.reindex(columns=self.meta["features"]))[0]), score)
        return score
