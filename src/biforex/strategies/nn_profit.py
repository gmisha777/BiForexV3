"""Strategy nn_profit: a neural network trained to maximize trading profit directly.

The network maps the lgbm_tb feature vector of a bar to a position p = tanh(...) in [-1, +1]
(+ long, - short, magnitude = conviction). Training does not predict a class: it adjusts the
weights by gradient ascent on the money the positions would have made under OUR exit rules,
    pnl_t = max(p_t, 0) * long_net_t + max(-p_t, 0) * short_net_t
    objective = mean(pnl) - risk_aversion * std(pnl)          (Sharpe-like)
where long_net / short_net are the triple-barrier trade results (take profit 2 ATR, stop 1 ATR,
time limit) minus costs. Regularization against memorizing noise: small network, dropout,
weight decay, early stopping on the purged tail of the training window, average of several
seeds. Score = p / q * 0.10 with q calibrated on validation predictions, so |score| >= 0.10 on
about `top_share` of bars, like the other strategies.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..data.schema import timeframe_delta
from ..model.registry import MODELS_DIR
from ..model.walkforward import monthly_folds
from . import register
from .base import Strategy, WalkForwardScores, clip_score
from .lgbm_tb import LgbmTripleBarrier

log = logging.getLogger(__name__)
torch.set_num_threads(4)

DEFAULTS = {
    "hidden": [32, 16],
    "dropout": 0.2,
    "weight_decay": 1e-3,
    "lr": 1e-3,
    "max_epochs": 300,
    "patience": 25,
    "risk_aversion": 0.5,
    "cost": 0.001,  # per round trip (limit entry + mixed exits)
    "seeds": 5,
    "val_fraction": 0.2,
    "top_share": 0.06,
}


class ProfitNet(nn.Module):
    def __init__(self, n_in: int, hidden: list[int], dropout: float):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.tanh(self.net(x)).squeeze(-1)


def profit_objective(p, long_net, short_net, risk_aversion: float):
    pnl = torch.relu(p) * long_net + torch.relu(-p) * short_net
    return pnl.mean() - risk_aversion * pnl.std()


class Scaler:
    """Robust scaling fitted on training rows; NaN -> 0 (the median)."""

    def fit(self, X: pd.DataFrame) -> Scaler:
        self.columns = list(X.columns)
        self.med = X.median()
        self.iqr = (X.quantile(0.75) - X.quantile(0.25)).replace(0, 1).fillna(1)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = (X[self.columns].astype("float64") - self.med) / self.iqr
        return Z.fillna(0).clip(-5, 5).to_numpy(dtype=np.float32)

    def state(self) -> dict:
        return {"columns": self.columns, "med": self.med.to_dict(), "iqr": self.iqr.to_dict()}

    @classmethod
    def from_state(cls, st: dict) -> Scaler:
        s = cls()
        s.columns, s.med, s.iqr = st["columns"], pd.Series(st["med"]), pd.Series(st["iqr"])
        return s


def train_net(Z_fit, ln_fit, sn_fit, Z_val, ln_val, sn_val, cfg: dict, seed: int):
    torch.manual_seed(seed)
    net = ProfitNet(Z_fit.shape[1], cfg["hidden"], cfg["dropout"])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    xf, lf, sf = map(torch.from_numpy, (Z_fit, ln_fit, sn_fit))
    xv, lv, sv = map(torch.from_numpy, (Z_val, ln_val, sn_val))
    best, best_state, bad = -np.inf, None, 0
    for _ in range(cfg["max_epochs"]):
        net.train()
        opt.zero_grad()
        loss = -profit_objective(net(xf), lf, sf, cfg["risk_aversion"])
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            val = profit_objective(net(xv), lv, sv, cfg["risk_aversion"]).item()
        if val > best:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, best


def predict(nets: list[nn.Module], Z: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(Z)
        return np.mean([n(x).numpy() for n in nets], axis=0)


class NetEnsemble:
    """Seeds of ProfitNet + scaler + score scale, fitted on one training window."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def fit(self, X: pd.DataFrame, long_ret: pd.Series, short_ret: pd.Series,
            purge: pd.Timedelta) -> NetEnsemble:
        cfg = self.cfg
        val_start = X.index[int(len(X) * (1 - cfg["val_fraction"]))]
        fit_m, val_m = X.index < val_start - purge, X.index >= val_start
        self.scaler = Scaler().fit(X[fit_m])
        Z = self.scaler.transform(X)
        ln = (long_ret - cfg["cost"]).to_numpy(np.float32)
        sn = (short_ret - cfg["cost"]).to_numpy(np.float32)
        self.nets, self.val = [], []
        for seed in range(cfg["seeds"]):
            net, v = train_net(Z[fit_m], ln[fit_m], sn[fit_m], Z[val_m], ln[val_m], sn[val_m],
                               cfg, seed)
            self.nets.append(net)
            self.val.append(v)
        p_val = predict(self.nets, Z[val_m])
        self.q = float(np.quantile(np.abs(p_val), 1 - cfg["top_share"])) or 1.0
        return self

    def positions(self, X: pd.DataFrame) -> np.ndarray:
        return predict(self.nets, self.scaler.transform(X))

    def scores(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.positions(X) / self.q * 0.10, -1, 1)


@register
class NnProfit(Strategy):
    name = "nn_profit"

    def __init__(self, settings, params: dict | None = None):
        super().__init__(settings, params)
        self.cfg = {**DEFAULTS, **self.params}
        self.features_src = LgbmTripleBarrier(settings)
        self.model: NetEnsemble | None = None
        self.meta: dict = {}

    def _dataset(self):
        (ds, symbols), = self.features_src._jobs().values()
        return ds, symbols

    def _purge(self) -> pd.Timedelta:
        st = self.settings
        return (st.label.horizon_bars + 1) * timeframe_delta(st.timeframe)

    def walk_forward(self) -> WalkForwardScores:
        st = self.settings
        ds, _ = self._dataset()
        purge = self._purge()
        out, folds = [], []
        for k, fold in enumerate(monthly_folds(ds.X.index, st.retrain.train_months, purge)):
            tr = (ds.X.index >= fold.train_start) & (ds.X.index < fold.train_end)
            te = (ds.X.index >= fold.test_start) & (ds.X.index < fold.test_end)
            if te.sum() == 0:
                continue
            o = ds.outcomes
            m = NetEnsemble(self.cfg).fit(ds.X[tr], o.loc[tr, "long_ret"], o.loc[tr, "short_ret"],
                                          purge)
            p = pd.DataFrame({"symbol": o.loc[te, "symbol"] if "symbol" in o else None,
                              "score": m.scores(ds.X[te])}, index=ds.X.index[te])
            out.append(p)
            folds.append({"fold": k, "test_start": fold.test_start, "q": m.q,
                          "val_objective": float(np.mean(m.val))})
        scores = pd.concat(out).sort_index(kind="stable")
        folds = pd.DataFrame(folds).set_index("fold")
        summary = (f"{len(folds)} folds x {self.cfg['seeds']} seeds; validation objective "
                   f"(mean pnl - {self.cfg['risk_aversion']} std) median "
                   f"{folds['val_objective'].median():.5f}; score std {scores['score'].std():.3f}")
        return WalkForwardScores(scores, {"folds": folds}, summary)

    def fit_final(self) -> dict:
        st = self.settings
        ds, symbols = self._dataset()
        start = ds.X.index[-1] - pd.DateOffset(months=st.retrain.train_months)
        mask = np.asarray(ds.X.index > start)
        o = ds.outcomes[mask]
        model = NetEnsemble(self.cfg).fit(ds.X[mask], o["long_ret"], o["short_ret"],
                                          self._purge())
        version = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        folder = MODELS_DIR / self.name / version
        folder.mkdir(parents=True, exist_ok=True)
        torch.save([n.state_dict() for n in model.nets], folder / "nets.pt")
        meta = {"name": self.name, "version": version, "timeframe": st.timeframe,
                "trained_until": str(ds.X.index[-1]), "symbols": symbols, "q": model.q,
                "cfg": self.cfg, "scaler": model.scaler.state()}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2, default=float),
                                          encoding="utf-8")
        (MODELS_DIR / self.name / "current.json").write_text(json.dumps({"version": version}),
                                                            encoding="utf-8")
        log.info("%s: saved to %s", self.name, folder)
        return meta

    def load(self) -> None:
        version = json.loads((MODELS_DIR / self.name / "current.json").read_text())["version"]
        folder = MODELS_DIR / self.name / version
        self.meta = json.loads((folder / "meta.json").read_text())
        if self.meta["timeframe"] != self.settings.timeframe:
            raise RuntimeError(f"{self.name}: trained on {self.meta['timeframe']}; retrain")
        cfg = self.meta["cfg"]
        model = NetEnsemble(cfg)
        model.scaler = Scaler.from_state(self.meta["scaler"])
        model.q = self.meta["q"]
        model.nets = []
        for state in torch.load(folder / "nets.pt"):
            net = ProfitNet(len(model.scaler.columns), cfg["hidden"], cfg["dropout"])
            net.load_state_dict(state)
            net.eval()
            model.nets.append(net)
        self.model = model
        log.info("%s/%s: trained until %s", self.name, version, self.meta["trained_until"])

    def score(self, symbol: str, bar: pd.Timestamp) -> float | None:
        f = self.features_src.features(symbol, live=True)
        if bar not in f.index:
            return None
        row = f.loc[[bar]].copy()
        row["symbol_id"] = np.float32(self.meta["symbols"].index(symbol))
        X = row.reindex(columns=self.model.scaler.columns)
        score = clip_score(float(self.model.scores(X)[0]))
        log.info("%s %s bar %s | score %+.3f", self.name, symbol, bar, score)
        return score
