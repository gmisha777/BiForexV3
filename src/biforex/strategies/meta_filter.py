"""Strategy meta_filter: decides how much to trust each lgbm_tb signal (meta-labeling).

lgbm_tb says where to bet; this model answers a different question: will this particular
signal reach its take profit? It is trained only on bars where lgbm_tb signalled
(|score| >= event_score), using lgbm_tb's out-of-sample scores, with the target "the trade
in the signal's direction hit take profit first". Features describe the context of the
signal: its strength, whether trend / flow / positioning agree with its direction, the
volatility regime, the calendar. A plain regularized logistic regression: there are only a
few hundred training signals.

score = lgbm_tb score x (p_success / average training success rate): signals the filter
distrusts shrink below the entry level, trusted ones grow. Use it INSTEAD of lgbm_tb in the
ensemble (it already contains lgbm_tb's opinion).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data.schema import timeframe_delta
from ..model.registry import MODELS_DIR
from . import register
from .base import Strategy, WalkForwardScores, clip_score
from .lgbm_tb import LgbmTripleBarrier

log = logging.getLogger(__name__)

DEFAULTS = {
    "event_score": 0.10,  # lgbm_tb |score| that counts as a signal
    "min_train_months": 12,  # of signals before the first out-of-sample month
    "C": 0.3,  # inverse regularization of the logistic regression
    "max_factor": 2.0,  # cap of p / mean_p
}
ALIGNED = ["ema55_dist_atr", "ema144_slope_atr", "h1d_ret_3", "ret_8", "ret_34",
           "macd_hist_atr", "taker_imbalance24", "funding_sum21", "global_ls_log",
           "oi_chg_24", "h1d_rsi14", "rsi14"]  # multiplied by the signal side
REGIME = ["atr_pct", "rv_ratio", "bb_width", "vol_rel24", "dow_sin", "dow_cos", "hour_sin",
          "hour_cos"]


def meta_features(features: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    """features: lgbm_tb features of the signal bars; score: lgbm_tb score (same index)."""
    side = np.sign(score)
    f = pd.DataFrame(index=features.index)
    f["abs_score"] = score.abs()
    f["side"] = side
    for c in ALIGNED:
        if c in features:
            x = features[c].astype("float64")
            if c.endswith("rsi14"):
                x = x - 50
            f[f"al_{c}"] = side * x
    if "range_pos168" in features:
        f["al_range_pos"] = side * (features["range_pos168"] - 0.5)
    for c in REGIME:
        if c in features:
            f[c] = features[c].astype("float64")
    return f


def new_model(C: float):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000))


@register
class MetaFilter(Strategy):
    name = "meta_filter"

    def __init__(self, settings, params: dict | None = None):
        super().__init__(settings, params)
        self.cfg = {**DEFAULTS, **self.params}
        self.base = LgbmTripleBarrier(settings)
        self.model = None
        self.mean_p = None

    # --- data --------------------------------------------------------------------------
    def _events(self) -> pd.DataFrame:
        """lgbm_tb out-of-sample signals with context features and outcomes."""
        path = self.base.report_dir / "oos_predictions.parquet"
        if not path.exists():
            raise FileNotFoundError("run scripts/train.py --strategies lgbm_tb first")
        p = pd.read_parquet(path)
        p = p[p["symbol"].isin(self.settings.symbols)]
        p = p[p["score"].abs() >= self.cfg["event_score"]]
        bar = timeframe_delta(self.settings.timeframe)
        parts = []
        for symbol, g in p.groupby("symbol"):
            feats = self.base.features(symbol).reindex(g.index)
            m = meta_features(feats, g["score"])
            long = g["score"] > 0
            m["success"] = np.where(long, g["long_outcome"] == 1, g["short_outcome"] == 1)
            held = np.where(long, g["long_bars"], g["short_bars"])
            m["known_at"] = g.index + (pd.Series(held, index=g.index) + 1) * bar
            m["symbol_id"] = float(self.settings.symbols.index(symbol))
            m["symbol"] = symbol
            m["lgbm_score"] = g["score"]
            parts.append(m)
        return pd.concat(parts).sort_index(kind="stable")

    @staticmethod
    def _xy(ev: pd.DataFrame):
        X = ev.drop(columns=["success", "known_at", "symbol", "lgbm_score"])
        return X, ev["success"].astype(int)

    def _factor(self, p: np.ndarray, mean_p: float) -> np.ndarray:
        return np.clip(p / mean_p, 0, self.cfg["max_factor"])

    # --- research ------------------------------------------------------------------------
    def walk_forward(self) -> WalkForwardScores:
        ev = self._events()
        X, y = self._xy(ev)
        first = (ev.index[0].normalize().replace(day=1)
                 + pd.DateOffset(months=self.cfg["min_train_months"]))
        lgbm = pd.read_parquet(self.base.report_dir / "oos_scores.parquet")
        lgbm = lgbm[lgbm["symbol"].isin(self.settings.symbols)]
        out = lgbm[lgbm.index >= first].copy()  # non-signal bars keep lgbm_tb's score
        rows, tested, month = [], [], first
        while month <= ev.index[-1]:
            nxt = month + pd.DateOffset(months=1)
            train = ev["known_at"] <= month  # outcome known before the test month starts
            test = (ev.index >= month) & (ev.index < nxt)
            if test.any() and train.sum() >= 50 and y[train].nunique() == 2:
                model = new_model(self.cfg["C"]).fit(X[train], y[train])
                mean_p = y[train].mean()
                p = model.predict_proba(X[test])[:, 1]
                factor = self._factor(p, mean_p)
                t = ev[test]
                key = pd.MultiIndex.from_arrays([t.index, t["symbol"]])
                adj = pd.Series(t["lgbm_score"].to_numpy() * factor, index=key)
                okey = pd.MultiIndex.from_arrays([out.index, out["symbol"]])
                hit = okey.isin(key)
                out.loc[hit, "score"] = adj.reindex(okey[hit]).to_numpy()
                tested.append(pd.DataFrame({"factor": factor, "success": y[test].to_numpy()}))
                rows.append({"month": month, "train_signals": int(train.sum()),
                             "test_signals": int(test.sum()), "mean_p_train": mean_p,
                             "kept": float((factor >= 1).mean()),
                             "success_kept": float(y[test][factor >= 1].mean())
                             if (factor >= 1).any() else np.nan,
                             "success_dropped": float(y[test][factor < 1].mean())
                             if (factor < 1).any() else np.nan})
            month = nxt
        folds = pd.DataFrame(rows).set_index("month")
        out["score"] = out["score"].clip(-1, 1)
        coef = pd.Series(new_model(self.cfg["C"]).fit(X, y)[-1].coef_[0], index=X.columns)
        t = pd.concat(tested)
        kept, dropped = t[t["factor"] >= 1], t[t["factor"] < 1]
        summary = (
            f"signals {len(ev)} ({ev.index[0].date()} -> {ev.index[-1].date()}), success rate "
            f"{y.mean():.3f}; out-of-sample from {first.date()}: {len(t)} signals\n"
            f"take-profit rate of signals the filter trusts: {kept['success'].mean():.3f} "
            f"({len(kept)}), distrusts: {dropped['success'].mean():.3f} ({len(dropped)})\n"
            f"standardized coefficients (all data, for insight):\n"
            f"{coef.sort_values(key=abs, ascending=False).round(3).head(12).to_string()}")
        return WalkForwardScores(out[["symbol", "score"]], {"folds": folds}, summary)

    def fit_final(self) -> dict:
        ev = self._events()
        X, y = self._xy(ev)
        model = new_model(self.cfg["C"]).fit(X, y)
        version = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        folder = MODELS_DIR / self.name / version
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "mean_p": float(y.mean()), "columns": list(X.columns)},
                    folder / "model.pkl")
        meta = {"name": self.name, "version": version, "timeframe": self.settings.timeframe,
                "signals": len(ev), "trained_until": str(ev.index[-1])}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (MODELS_DIR / self.name / "current.json").write_text(json.dumps({"version": version}),
                                                            encoding="utf-8")
        log.info("%s: saved to %s (%d signals)", self.name, folder, len(ev))
        return meta

    # --- live ------------------------------------------------------------------------------
    def load(self) -> None:
        self.base.load()
        version = json.loads((MODELS_DIR / self.name / "current.json").read_text())["version"]
        blob = joblib.load(MODELS_DIR / self.name / version / "model.pkl")
        self.model, self.mean_p, self.columns = blob["model"], blob["mean_p"], blob["columns"]
        log.info("%s/%s loaded (mean success %.3f)", self.name, version, self.mean_p)

    def score(self, symbol: str, bar: pd.Timestamp) -> float | None:
        base = self.base.score(symbol, bar)
        if base is None or abs(base) < self.cfg["event_score"]:
            return base
        feats = self.base.features(symbol, live=True).loc[[bar]]
        X = meta_features(feats, pd.Series([base], index=feats.index))
        X["symbol_id"] = float(self.settings.symbols.index(symbol))
        p = self.model.predict_proba(X.reindex(columns=self.columns))[0, 1]
        score = clip_score(base * float(self._factor(np.array([p]), self.mean_p)[0]))
        log.info("%s %s bar %s | lgbm %+.3f, p_success %.3f (mean %.3f) -> %+.3f", self.name,
                 symbol, bar, base, p, self.mean_p, score)
        return score
