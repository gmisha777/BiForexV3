"""Strategy nn_seq: a 1D-CNN on raw bar sequences, trained to maximize trade profit directly.

Instead of hand-made features, the network sees the last `window` bars (per bar: move in ATR,
range, body and wicks, relative volume, taker buy share, open-interest change, funding,
volatility ratio, time of day / week) and learns its own patterns. Output, objective, costs,
regularization and score calibration are those of nn_profit. Retrained every
`retrain_every` months to keep the walk-forward affordable on a CPU.
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
from .nn_profit import DEFAULTS as NN_DEFAULTS
from .nn_profit import profit_objective

log = logging.getLogger(__name__)

DEFAULTS = {**NN_DEFAULTS, "window": 24, "channels": 16, "retrain_every": 3, "seeds": 3,
            "batch": 4096, "max_epochs": 150, "patience": 15}
STEP_COLUMNS = ["range_atr", "body_ratio", "upper_wick", "lower_wick", "vol_rel24",
                "taker_buy_ratio", "oi_chg_1", "funding_last", "rv_ratio", "hour_sin",
                "hour_cos", "dow_sin", "dow_cos"]


def step_features(f: pd.DataFrame) -> pd.DataFrame:
    """Per-bar inputs of the sequence model from the lgbm_tb feature table."""
    s = pd.DataFrame(index=f.index)
    s["move_atr"] = f["ret_1"] / f["atr_pct"]
    for c in STEP_COLUMNS:
        s[c] = f[c] if c in f else 0.0
    return s.astype("float32")


def windows(steps: np.ndarray, window: int) -> np.ndarray:
    """(n, c) -> (n, window, c); rows without enough history are zero-padded at the start."""
    pad = np.zeros((window - 1, steps.shape[1]), dtype=np.float32)
    x = np.concatenate([pad, steps])
    return np.lib.stride_tricks.sliding_window_view(x, window, axis=0).transpose(0, 2, 1).copy()


class SeqNet(nn.Module):
    def __init__(self, n_in: int, channels: int, dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_in, channels, 3, padding=1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=2, dilation=2), nn.GELU(),
            nn.Dropout(dropout))
        self.head = nn.Linear(2 * channels, 1)

    def forward(self, x):  # x: (batch, window, features)
        h = self.conv(x.transpose(1, 2))  # (batch, channels, window)
        pooled = torch.cat([h.mean(-1), h[..., -1]], dim=1)
        return torch.tanh(self.head(pooled)).squeeze(-1)


class SeqEnsemble:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _scale(self, S: np.ndarray) -> np.ndarray:
        return np.clip(np.nan_to_num((S - self.med) / self.iqr), -5, 5).astype(np.float32)

    def fit(self, S_rows: np.ndarray, W: np.ndarray, index: pd.DatetimeIndex, long_ret, short_ret,
            purge: pd.Timedelta) -> SeqEnsemble:
        """S_rows: last-step inputs of the training rows (for scaling); W: raw windows."""
        cfg = self.cfg
        self.med = np.nanmedian(S_rows, axis=0)
        q75, q25 = np.nanpercentile(S_rows, 75, axis=0), np.nanpercentile(S_rows, 25, axis=0)
        self.iqr = np.where((q75 - q25) > 0, q75 - q25, 1.0)
        X = self._scale(W)
        val_start = index[int(len(index) * (1 - cfg["val_fraction"]))]
        fit_m, val_m = np.asarray(index < val_start - purge), np.asarray(index >= val_start)
        ln = torch.from_numpy((np.asarray(long_ret) - cfg["cost"]).astype(np.float32))
        sn = torch.from_numpy((np.asarray(short_ret) - cfg["cost"]).astype(np.float32))
        xt = torch.from_numpy(X)
        self.nets = []
        for seed in range(cfg["seeds"]):
            torch.manual_seed(seed)
            net = SeqNet(X.shape[2], cfg["channels"], cfg["dropout"])
            opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"],
                                    weight_decay=cfg["weight_decay"])
            fit_idx = np.flatnonzero(fit_m)
            best, best_state, bad = -np.inf, None, 0
            rng = np.random.default_rng(seed)
            for _ in range(cfg["max_epochs"]):
                net.train()
                for b in np.array_split(rng.permutation(fit_idx),
                                        max(1, len(fit_idx) // cfg["batch"])):
                    opt.zero_grad()
                    loss = -profit_objective(net(xt[b]), ln[b], sn[b], cfg["risk_aversion"])
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    v = profit_objective(net(xt[val_m]), ln[val_m], sn[val_m],
                                         cfg["risk_aversion"]).item()
                if v > best:
                    best, bad = v, 0
                    best_state = {k: t.clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= cfg["patience"]:
                        break
            net.load_state_dict(best_state)
            net.eval()
            self.nets.append(net)
        p_val = self.positions(W[val_m])
        self.q = float(np.quantile(np.abs(p_val), 1 - cfg["top_share"])) or 1.0
        return self

    def positions(self, W: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(self._scale(W))
            return np.mean([n(x).numpy() for n in self.nets], axis=0)

    def scores(self, W: np.ndarray) -> np.ndarray:
        return np.clip(self.positions(W) / self.q * 0.10, -1, 1)


@register
class NnSeq(Strategy):
    name = "nn_seq"

    def __init__(self, settings, params: dict | None = None):
        super().__init__(settings, params)
        self.cfg = {**DEFAULTS, **self.params}
        self.features_src = LgbmTripleBarrier(settings)
        self.model: SeqEnsemble | None = None

    def _symbol_windows(self, symbol: str, live: bool = False):
        steps = step_features(self.features_src.features(symbol, live))
        return steps, windows(steps.to_numpy(), self.cfg["window"])

    def _data(self):
        (ds, symbols), = self.features_src._jobs().values()
        sym = ds.outcomes["symbol"].to_numpy()
        n_c = len(STEP_COLUMNS) + 1
        W = np.zeros((len(ds.X), self.cfg["window"], n_c), dtype=np.float32)
        for s in symbols:
            steps, w = self._symbol_windows(s)
            pos = steps.index.get_indexer(ds.X.index[sym == s])
            W[sym == s] = w[pos]
        return ds, W

    def walk_forward(self) -> WalkForwardScores:
        st = self.settings
        ds, W = self._data()
        purge = (st.label.horizon_bars + 1) * timeframe_delta(st.timeframe)
        out, model, rows = [], None, []
        for k, fold in enumerate(monthly_folds(ds.X.index, st.retrain.train_months, purge)):
            te = (ds.X.index >= fold.test_start) & (ds.X.index < fold.test_end)
            if te.sum() == 0:
                continue
            if model is None or k % self.cfg["retrain_every"] == 0:
                tr = (ds.X.index >= fold.train_start) & (ds.X.index < fold.train_end)
                o = ds.outcomes[tr]
                model = SeqEnsemble(self.cfg).fit(W[tr][:, -1, :], W[tr], ds.X.index[tr],
                                                  o["long_ret"], o["short_ret"], purge)
                rows.append({"fold": k, "test_start": fold.test_start, "q": model.q})
                log.info("%s: fold %d trained (%d rows)", self.name, k, tr.sum())
            out.append(pd.DataFrame({"symbol": ds.outcomes.loc[te, "symbol"],
                                     "score": model.scores(W[te])}, index=ds.X.index[te]))
        scores = pd.concat(out).sort_index(kind="stable")
        summary = f"{len(rows)} trainings x {self.cfg['seeds']} seeds; score std " \
                  f"{scores['score'].std():.3f}"
        return WalkForwardScores(scores, {"trainings": pd.DataFrame(rows)}, summary)

    def fit_final(self) -> dict:
        st = self.settings
        ds, W = self._data()
        start = ds.X.index[-1] - pd.DateOffset(months=st.retrain.train_months)
        m = np.asarray(ds.X.index > start)
        purge = (st.label.horizon_bars + 1) * timeframe_delta(st.timeframe)
        o = ds.outcomes[m]
        model = SeqEnsemble(self.cfg).fit(W[m][:, -1, :], W[m], ds.X.index[m], o["long_ret"],
                                          o["short_ret"], purge)
        version = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        folder = MODELS_DIR / self.name / version
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({"nets": [n.state_dict() for n in model.nets], "med": model.med,
                    "iqr": model.iqr, "q": model.q}, folder / "model.pt")
        meta = {"name": self.name, "version": version, "timeframe": st.timeframe,
                "trained_until": str(ds.X.index[-1]), "cfg": self.cfg}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2, default=float),
                                          encoding="utf-8")
        (MODELS_DIR / self.name / "current.json").write_text(json.dumps({"version": version}),
                                                            encoding="utf-8")
        return meta

    def load(self) -> None:
        version = json.loads((MODELS_DIR / self.name / "current.json").read_text())["version"]
        folder = MODELS_DIR / self.name / version
        meta = json.loads((folder / "meta.json").read_text())
        if meta["timeframe"] != self.settings.timeframe:
            raise RuntimeError(f"{self.name}: trained on {meta['timeframe']}; retrain")
        blob = torch.load(folder / "model.pt", weights_only=False)
        model = SeqEnsemble(meta["cfg"])
        model.med, model.iqr, model.q = blob["med"], blob["iqr"], blob["q"]
        model.nets = []
        for state in blob["nets"]:
            net = SeqNet(len(STEP_COLUMNS) + 1, meta["cfg"]["channels"], meta["cfg"]["dropout"])
            net.load_state_dict(state)
            net.eval()
            model.nets.append(net)
        self.model = model

    def score(self, symbol: str, bar: pd.Timestamp) -> float | None:
        steps, w = self._symbol_windows(symbol, live=True)
        if bar not in steps.index:
            return None
        i = steps.index.get_loc(bar)
        score = clip_score(float(self.model.scores(w[[i]])[0]))
        log.info("%s %s bar %s | score %+.3f", self.name, symbol, bar, score)
        return score
