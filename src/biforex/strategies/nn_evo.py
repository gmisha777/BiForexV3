"""Strategy nn_evo: a small neural network whose weights are evolved for trading profit.

Neuroevolution ("fitting the neurons' coefficients" without gradients): a population of
weight vectors around the current one is evaluated by a fast trade simulation on the training
window, and the weights move toward the better ones (evolution strategy, OpenAI-ES style).
Fitness is the Sharpe ratio of the trades the network would actually take: bars where
|score| >= entry_score, in the score's direction, with the triple-barrier results of our exit
rules (take profit / stop / time limit) minus costs. Unlike nn_profit this optimizes the
thresholded, non-differentiable trading rule itself. Early stopping on the purged tail of the
training window. Score = network output scaled like the other strategies.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ..data.schema import timeframe_delta
from ..model.registry import MODELS_DIR
from ..model.walkforward import monthly_folds
from . import register
from .base import Strategy, WalkForwardScores, clip_score
from .lgbm_tb import LgbmTripleBarrier
from .nn_profit import Scaler

log = logging.getLogger(__name__)

DEFAULTS = {
    "hidden": 8,
    "population": 32,  # antithetic pairs -> 2 x population evaluations per step
    "sigma": 0.05,
    "lr": 0.03,
    "iterations": 60,
    "patience": 10,
    "entry_score": 0.10,
    "min_trades": 30,  # fewer trades in a window -> fitness penalized
    "cost": 0.001,
    "val_fraction": 0.2,
    "seeds": 2,
    "retrain_every": 3,  # months; evolution is slow on a CPU
}


class EvoNet:
    """x -> tanh(W2 . gelu(W1 x + b1) + b2); weights as one flat vector."""

    def __init__(self, n_in: int, hidden: int):
        self.n_in, self.hidden = n_in, hidden
        self.size = n_in * hidden + hidden + hidden + 1

    def unpack(self, w: np.ndarray):
        i, h = self.n_in, self.hidden
        W1 = w[: i * h].reshape(i, h)
        b1 = w[i * h: i * h + h]
        W2 = w[i * h + h: i * h + 2 * h]
        b2 = w[-1]
        return W1, b1, W2, b2

    def forward(self, w: np.ndarray, Z: np.ndarray) -> np.ndarray:
        W1, b1, W2, b2 = self.unpack(w)
        a = Z @ W1 + b1
        a = 0.5 * a * (1 + np.tanh(0.7978845608 * (a + 0.044715 * a**3)))  # GELU
        return np.tanh(a @ W2 + b2)


def trade_sharpe(score: np.ndarray, long_net: np.ndarray, short_net: np.ndarray,
                 entry: float, min_trades: int) -> float:
    long_, short_ = score >= entry, score <= -entry
    pnl = np.concatenate([long_net[long_], short_net[short_]])
    if len(pnl) < 2:
        return -1.0
    sharpe = pnl.mean() / (pnl.std() + 1e-9) * np.sqrt(len(pnl))
    return float(sharpe * min(1.0, len(pnl) / min_trades))


class EvoModel:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _scores(self, w, Z):
        return self.net.forward(w, Z) * self.scale

    def fit(self, X: pd.DataFrame, long_ret: pd.Series, short_ret: pd.Series,
            purge: pd.Timedelta) -> EvoModel:
        cfg = self.cfg
        val_start = X.index[int(len(X) * (1 - cfg["val_fraction"]))]
        fit_m = np.asarray(X.index < val_start - purge)
        val_m = np.asarray(X.index >= val_start)
        self.scaler = Scaler().fit(X[fit_m])
        Z = self.scaler.transform(X)
        ln = long_ret.to_numpy() - cfg["cost"]
        sn = short_ret.to_numpy() - cfg["cost"]
        self.net = EvoNet(Z.shape[1], cfg["hidden"])
        self.scale = 0.3  # tanh output -> score; |score| >= 0.10 needs |tanh| >= 1/3
        self.weights = []
        for seed in range(cfg["seeds"]):
            rng = np.random.default_rng(seed)
            w = rng.normal(0, 0.1, self.net.size)
            best_w, best_val, bad = w.copy(), -np.inf, 0
            for _ in range(cfg["iterations"]):
                eps = rng.normal(0, 1, (cfg["population"], self.net.size))
                f_pos = np.array([trade_sharpe(self._scores(w + cfg["sigma"] * e, Z[fit_m]),
                                               ln[fit_m], sn[fit_m], cfg["entry_score"],
                                               cfg["min_trades"]) for e in eps])
                f_neg = np.array([trade_sharpe(self._scores(w - cfg["sigma"] * e, Z[fit_m]),
                                               ln[fit_m], sn[fit_m], cfg["entry_score"],
                                               cfg["min_trades"]) for e in eps])
                adv = f_pos - f_neg
                adv = adv / (adv.std() + 1e-9)
                w = w + cfg["lr"] / (2 * cfg["sigma"]) * (adv[:, None] * eps).mean(0)
                v = trade_sharpe(self._scores(w, Z[val_m]), ln[val_m], sn[val_m],
                                 cfg["entry_score"], cfg["min_trades"])
                if v > best_val:
                    best_w, best_val, bad = w.copy(), v, 0
                else:
                    bad += 1
                    if bad >= cfg["patience"]:
                        break
            self.weights.append(best_w)
        return self

    def scores(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.scaler.transform(X)
        return np.clip(np.mean([self._scores(w, Z) for w in self.weights], axis=0), -1, 1)


@register
class NnEvo(Strategy):
    name = "nn_evo"

    def __init__(self, settings, params: dict | None = None):
        super().__init__(settings, params)
        self.cfg = {**DEFAULTS, **self.params}
        self.features_src = LgbmTripleBarrier(settings)
        self.model: EvoModel | None = None
        self.meta: dict = {}

    def _dataset(self):
        (ds, symbols), = self.features_src._jobs().values()
        return ds, symbols

    def _purge(self):
        st = self.settings
        return (st.label.horizon_bars + 1) * timeframe_delta(st.timeframe)

    def walk_forward(self) -> WalkForwardScores:
        st = self.settings
        ds, _ = self._dataset()
        purge, out, m = self._purge(), [], None
        for k, fold in enumerate(monthly_folds(ds.X.index, st.retrain.train_months, purge)):
            te = (ds.X.index >= fold.test_start) & (ds.X.index < fold.test_end)
            if te.sum() == 0:
                continue
            o = ds.outcomes
            if m is None or k % self.cfg["retrain_every"] == 0:
                tr = (ds.X.index >= fold.train_start) & (ds.X.index < fold.train_end)
                m = EvoModel(self.cfg).fit(ds.X[tr], o.loc[tr, "long_ret"],
                                           o.loc[tr, "short_ret"], purge)
                log.info("%s: fold %d evolved", self.name, k)
            out.append(pd.DataFrame({"symbol": o.loc[te, "symbol"], "score": m.scores(ds.X[te])},
                                    index=ds.X.index[te]))
        scores = pd.concat(out).sort_index(kind="stable")
        return WalkForwardScores(scores, {}, f"score std {scores['score'].std():.3f}, "
                                             f"|score|>=0.10 on "
                                             f"{100 * (scores['score'].abs() >= 0.1).mean():.1f}%")

    def fit_final(self) -> dict:
        st = self.settings
        ds, symbols = self._dataset()
        start = ds.X.index[-1] - pd.DateOffset(months=st.retrain.train_months)
        m = np.asarray(ds.X.index > start)
        o = ds.outcomes[m]
        model = EvoModel(self.cfg).fit(ds.X[m], o["long_ret"], o["short_ret"], self._purge())
        version = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        folder = MODELS_DIR / self.name / version
        folder.mkdir(parents=True, exist_ok=True)
        meta = {"name": self.name, "version": version, "timeframe": st.timeframe,
                "trained_until": str(ds.X.index[-1]), "symbols": symbols, "cfg": self.cfg,
                "scaler": model.scaler.state(), "weights": [w.tolist() for w in model.weights]}
        (folder / "meta.json").write_text(json.dumps(meta, default=float), encoding="utf-8")
        (MODELS_DIR / self.name / "current.json").write_text(json.dumps({"version": version}),
                                                            encoding="utf-8")
        return {k: v for k, v in meta.items() if k != "weights"}

    def load(self) -> None:
        version = json.loads((MODELS_DIR / self.name / "current.json").read_text())["version"]
        self.meta = json.loads((MODELS_DIR / self.name / version / "meta.json").read_text())
        if self.meta["timeframe"] != self.settings.timeframe:
            raise RuntimeError(f"{self.name}: trained on {self.meta['timeframe']}; retrain")
        m = EvoModel(self.meta["cfg"])
        m.scaler = Scaler.from_state(self.meta["scaler"])
        m.net = EvoNet(len(m.scaler.columns), self.meta["cfg"]["hidden"])
        m.scale = 0.3
        m.weights = [np.array(w) for w in self.meta["weights"]]
        self.model = m

    def score(self, symbol: str, bar: pd.Timestamp) -> float | None:
        f = self.features_src.features(symbol, live=True)
        if bar not in f.index:
            return None
        row = f.loc[[bar]].copy()
        row["symbol_id"] = np.float32(self.meta["symbols"].index(symbol))
        score = clip_score(float(self.model.scores(row.reindex(
            columns=self.model.scaler.columns))[0]))
        log.info("%s %s bar %s | score %+.3f", self.name, symbol, bar, score)
        return score
