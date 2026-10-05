"""Strategy nn_slope: a neural network forecasts the trend slope of the next bars.

Input: the last `window` (24) bars, scale-free - closes relative to the last close, wicks and
bodies, all in ATR(14) of the last bar; volume relative to the window mean; taker buy share;
plus the volatility level (ATR / close) and the hour of day.
Target: the least-squares slope of the closes over the next `horizon` (12) bars, in ATR per
bar ("slope"), and the same multiplied by the fit's R^2 ("slope_r2": a straight trend counts
more than a jagged one). One network predicts both.
Networks: "mlp" on the flattened window, or "cnn" (1D convolutions over the bars); `seeds`
copies averaged. Trained on all symbols (data.extra_symbols too), traded on `symbols`.

Walk-forward: retrained every `retrain_every` months on up to `train_months` of earlier
data (at least `min_train_months`), so forecasts exist from early 2021. Forecasts before
`calibration_end` choose the network, the target, the share of bars that trade
(`top_share` -> score scale), the stop and the exit level by in-sample backtests (exits:
stop, an opposite signal, or a `max_hold` safety limit; no take profit). Saved scores start
at `calibration_end`.
"""

from __future__ import annotations

import itertools
import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..data import store
from ..data.schema import timeframe_delta
from ..features.indicators import atr as atr_fn
from ..model.registry import MODELS_DIR
from . import register
from .base import Strategy, WalkForwardScores, clip_score

log = logging.getLogger(__name__)

DEFAULTS = {
    "window": 24, "horizon": 12, "seeds": 2, "hidden": [64, 32], "channels": 32,
    "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-3, "batch": 2048, "max_epochs": 30,
    "patience": 4, "val_fraction": 0.15,
    "train_months": 36, "min_train_months": 12, "retrain_every": 3,
    "calibration_end": "2023-04-01",
    # choices made on in-sample forecasts
    "model": ["mlp", "cnn"], "target": ["slope", "slope_r2"], "top_share": [0.02, 0.05, 0.10],
    "sl_atr": [1.0, 1.5, 2.0], "exit_score": [0.10, 0.03], "max_hold": 48,
    "min_in_sample_trades": 50,
}
TARGETS = ["slope", "slope_r2"]
N_CH = 6  # per-bar channels
N_SCALAR = 3


# --- data -------------------------------------------------------------------------------------
class SymbolBars:
    def __init__(self, k: pd.DataFrame, window: int, horizon: int, bar: pd.Timedelta):
        k = k.sort_values("open_time").reset_index(drop=True)
        self.time = pd.DatetimeIndex(k["open_time"])
        o, h, lo, c = (k[x].to_numpy(np.float64) for x in ("open", "high", "low", "close"))
        v = k["volume"].to_numpy(np.float64)
        a = atr_fn(k["high"], k["low"], k["close"]).to_numpy()
        self.c, self.atr = c, a
        n = len(k)
        self.lv = np.log1p(v)
        buy = k["taker_buy_volume"].to_numpy(np.float64)
        self.imb = np.where(v > 0, 2 * buy / np.where(v > 0, v, 1) - 1, 0.0)
        self.o, self.h, self.lo = o, h, lo
        t_ns = self.time.as_unit("ns").asi8
        step = bar.value
        # rows with a contiguous input window and a valid ATR
        ok = np.zeros(n, bool)
        ok[window - 1:] = (t_ns[window - 1:] - t_ns[:n - window + 1]) == (window - 1) * step
        ok &= np.isfinite(a) & (a > 0)
        self.ok = ok
        self.hour = self.time.hour.to_numpy()
        # targets
        self.y = np.full((n, 2), np.nan, np.float32)
        m = n - horizon
        if m > 0:
            idx = np.arange(m)
            fut = idx[:, None] + np.arange(0, horizon + 1)[None, :]
            cont = (t_ns[idx + horizon] - t_ns[idx]) == horizon * step
            path = (c[fut] - c[idx][:, None]) / a[idx][:, None]
            x = np.arange(horizon + 1, dtype=np.float64)
            xc = x - x.mean()
            slope = (path - path.mean(1, keepdims=True)) @ xc / (xc @ xc)
            fit = slope[:, None] * xc[None, :] + path.mean(1, keepdims=True)
            ss_res = ((path - fit) ** 2).sum(1)
            ss_tot = ((path - path.mean(1, keepdims=True)) ** 2).sum(1)
            r2 = np.where(ss_tot > 0, 1 - ss_res / np.where(ss_tot > 0, ss_tot, 1), 0.0)
            good = cont & ok[idx]
            self.y[idx[good], 0] = slope[good]
            self.y[idx[good], 1] = slope[good] * r2[good]
        self.window = window

    def inputs(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(n, N_CH, window) per-bar channels and (n, N_SCALAR) scalars."""
        w = self.window
        idx = rows[:, None] + np.arange(1 - w, 1)[None, :]
        a = self.atr[rows][:, None]
        c_last = self.c[rows][:, None]
        c, o, h, lo = self.c[idx], self.o[idx], self.h[idx], self.lo[idx]
        x = np.empty((len(rows), N_CH, w), np.float32)
        x[:, 0] = (c - c_last) / a
        x[:, 1] = (h - np.maximum(o, c)) / a
        x[:, 2] = (np.minimum(o, c) - lo) / a
        x[:, 3] = (c - o) / a
        lv = self.lv[idx]
        x[:, 4] = lv - lv.mean(1, keepdims=True)
        x[:, 5] = self.imb[idx]
        np.clip(x, -20, 20, out=x)
        hr = self.hour[rows] * (2 * np.pi / 24)
        s = np.stack([np.log(self.atr[rows] / self.c[rows]) + 5, np.sin(hr), np.cos(hr)], 1)
        return x, s.astype(np.float32)


# --- networks ---------------------------------------------------------------------------------
class MlpNet(nn.Module):
    def __init__(self, window: int, hidden: list[int], dropout: float, **_):
        super().__init__()
        layers, prev = [], N_CH * window + N_SCALAR
        for hd in hidden:
            layers += [nn.Linear(prev, hd), nn.GELU(), nn.Dropout(dropout)]
            prev = hd
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, len(TARGETS))

    def forward(self, x, s):
        return self.head(self.body(torch.cat([x.flatten(1), s], 1)))


class CnnNet(nn.Module):
    def __init__(self, window: int, hidden: list[int], dropout: float, channels: int, **_):
        super().__init__()
        ch = channels
        self.conv = nn.Sequential(
            nn.Conv1d(N_CH, ch, 3, padding=1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(ch, ch, 3, padding=2, dilation=2), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(ch, ch, 3, padding=4, dilation=4), nn.GELU())
        self.join = nn.Sequential(nn.Linear(2 * ch + N_SCALAR, hidden[-1]), nn.GELU(),
                                  nn.Dropout(dropout))
        self.head = nn.Linear(hidden[-1], len(TARGETS))

    def forward(self, x, s):
        h = self.conv(x)
        return self.head(self.join(torch.cat([h.mean(-1), h[..., -1], s], 1)))


NETS = {"mlp": MlpNet, "cnn": CnnNet}


class SlopeModel:
    """`seeds` networks of one type trained on the same rows; forecasts averaged."""

    def __init__(self, cfg: dict, kind: str):
        self.cfg, self.kind = cfg, kind
        self.nets: list[nn.Module] = []

    def _net(self):
        c = self.cfg
        return NETS[self.kind](window=c["window"], hidden=c["hidden"], dropout=c["dropout"],
                               channels=c["channels"])

    def fit(self, data: list[SymbolBars], rows: list[np.ndarray], purge_ns: int) -> SlopeModel:
        cfg = self.cfg
        sid = np.concatenate([np.full(len(r), k) for k, r in enumerate(rows)])
        row = np.concatenate(rows)
        t = np.concatenate([d.time[r].as_unit("ns").asi8 for d, r in zip(data, rows, strict=True)])
        Y = np.concatenate([d.y[r] for d, r in zip(data, rows, strict=True)])
        cut = np.quantile(t, 1 - cfg["val_fraction"])
        fit_i, val_i = np.flatnonzero(t < cut - purge_ns), np.flatnonzero(t >= cut)
        self.y_std = Y[fit_i].std(0) + 1e-6
        yt = torch.from_numpy((Y / self.y_std).astype(np.float32))

        def batch(b):
            x = np.empty((len(b), N_CH, cfg["window"]), np.float32)
            s = np.empty((len(b), N_SCALAR), np.float32)
            for k in np.unique(sid[b]):
                m = sid[b] == k
                x[m], s[m] = data[k].inputs(row[b][m])
            return torch.from_numpy(x), torch.from_numpy(s)

        vb = [(batch(b), yt[b]) for b in np.array_split(val_i, max(1, len(val_i) // 8192))]
        loss_fn = nn.HuberLoss()
        self.nets, self.val_loss = [], []
        for seed in range(cfg["seeds"]):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            net = self._net()
            opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"],
                                    weight_decay=cfg["weight_decay"])
            best, state, bad = np.inf, None, 0
            for _ in range(cfg["max_epochs"]):
                net.train()
                for b in np.array_split(rng.permutation(fit_i),
                                        max(1, len(fit_i) // cfg["batch"])):
                    x, s = batch(b)
                    opt.zero_grad()
                    loss = loss_fn(net(x, s), yt[b])
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    v = float(np.mean([loss_fn(net(*xb), yb).item() for xb, yb in vb]))
                if v < best - 1e-5:
                    best, bad = v, 0
                    state = {k: x.clone() for k, x in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= cfg["patience"]:
                        break
            net.load_state_dict(state)
            net.eval()
            self.nets.append(net)
            self.val_loss.append(best)
        p = np.concatenate([self._predict(*xb) for xb, _ in vb])
        yv = Y[val_i]
        self.val_corr = [float(np.corrcoef(p[:, j], yv[:, j])[0, 1]) for j in range(2)]
        return self

    def _predict(self, x, s) -> np.ndarray:
        with torch.no_grad():
            return np.mean([n(x, s).numpy() for n in self.nets], axis=0) * self.y_std

    def predict(self, d: SymbolBars, rows: np.ndarray) -> np.ndarray:
        out = [self._predict(*map(torch.from_numpy, d.inputs(b)))
               for b in np.array_split(rows, max(1, len(rows) // 8192)) if len(b)]
        return np.concatenate(out) if out else np.empty((0, 2), np.float32)

    def state(self) -> dict:
        return {"kind": self.kind, "cfg": self.cfg, "y_std": self.y_std,
                "nets": [n.state_dict() for n in self.nets]}

    @classmethod
    def from_state(cls, blob: dict) -> SlopeModel:
        m = cls(blob["cfg"], blob["kind"])
        m.y_std = blob["y_std"]
        for st in blob["nets"]:
            net = m._net()
            net.load_state_dict(st)
            net.eval()
            m.nets.append(net)
        return m


# --- strategy ---------------------------------------------------------------------------------
@register
class NnSlope(Strategy):
    name = "nn_slope"

    def __init__(self, settings, params: dict | None = None):
        super().__init__(settings, params)
        self.cfg = {**DEFAULTS, **self.params}
        self.cal_end = pd.Timestamp(self.cfg["calibration_end"], tz="UTC")
        self.bar = timeframe_delta(settings.timeframe)
        self.chosen: dict = {}
        self.model: SlopeModel | None = None

    def _bars(self, symbol: str, live: bool = False) -> SymbolBars:
        k = store.load(store.klines_path(self.settings.data.dir, symbol, self.settings.timeframe))
        if live:
            k = k.iloc[-(self.cfg["window"] + 200):]
        return SymbolBars(k, self.cfg["window"], self.cfg["horizon"], self.bar)

    def _symbols(self) -> list[str]:
        d, tf = self.settings.data.dir, self.settings.timeframe
        return [s for s in self.settings.all_symbols if store.klines_path(d, s, tf).exists()]

    # --- walk-forward forecasts ---------------------------------------------------------------
    def forecasts(self) -> pd.DataFrame:
        """Out-of-sample forecasts of both networks for the traded symbols: columns symbol,
        <model>_<target>."""
        cfg = self.cfg
        path = self.report_dir / "oos_forecasts.parquet"
        if path.exists():
            return pd.read_parquet(path)
        syms = self._symbols()
        data = [self._bars(s) for s in syms]
        traded = [k for k, s in enumerate(syms) if s in self.settings.symbols]
        purge = (cfg["horizon"] + 1) * self.bar
        first = min(d.time[0] for d in data).replace(day=1, hour=0, minute=0, second=0)
        start = first + pd.DateOffset(months=cfg["min_train_months"])
        last = max(d.time[-1] for d in data)
        out, rows_log = {s: {} for s in self.settings.symbols}, []
        while start <= last:
            end = start + pd.DateOffset(months=cfg["retrain_every"])
            train_lo = start - pd.DateOffset(months=cfg["train_months"])
            rows = []
            for d in data:
                m = d.ok & np.isfinite(d.y[:, 0]) & (d.time >= train_lo) & (
                    d.time + (cfg["horizon"] + 1) * self.bar <= start)
                rows.append(np.flatnonzero(m))
            for kind in cfg["model"]:
                model = SlopeModel(cfg, kind).fit(data, rows, purge.value)
                rows_log.append({"test_start": start.date(), "model": kind,
                                 "train_rows": sum(len(r) for r in rows),
                                 "val_corr_slope": round(model.val_corr[0], 4),
                                 "val_corr_slope_r2": round(model.val_corr[1], 4)})
                log.info("%s: %s", self.name, rows_log[-1])
                for k in traded:
                    d = data[k]
                    te = np.flatnonzero(d.ok & (d.time >= start) & (d.time < end))
                    if len(te):
                        p = model.predict(d, te)
                        for j, tgt in enumerate(TARGETS):
                            out[syms[k]].setdefault(f"{kind}_{tgt}", []).append(
                                pd.Series(p[:, j], index=d.time[te]))
            start = end
        parts = []
        for s, cols in out.items():
            df = pd.DataFrame({c: pd.concat(v) for c, v in cols.items()})
            df.insert(0, "symbol", s)
            parts.append(df)
        fc = pd.concat(parts).sort_index(kind="stable")
        fc.index.name = "open_time"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        fc.to_parquet(path)
        pd.DataFrame(rows_log).to_csv(self.report_dir / "trainings.csv", index=False)
        return fc

    def _score_frame(self, fc: pd.DataFrame, col: str, share: float) -> tuple[pd.DataFrame, dict]:
        parts, q = [], {}
        for s, g in fc.groupby("symbol"):
            ins = g.loc[g.index < self.cal_end, col].dropna()
            q[s] = float(ins.abs().quantile(1 - share)) or 1.0
            parts.append(pd.DataFrame({"symbol": s,
                                       "score": (g[col] / q[s] * 0.10).clip(-1, 1)}))
        return pd.concat(parts).sort_index(kind="stable").dropna(), q

    def backtest(self, scores: pd.DataFrame, sl: float, exit_score: float, equity=None,
                 rules=None):
        from ..backtest import metrics
        from ..backtest.engine import params_from, run_backtest
        from ..labels.build import labels_path

        st, d = self.settings, self.settings.data.dir
        parts = []
        for s, g in scores.groupby("symbol"):
            atr = store.load(labels_path(st, s)).set_index("open_time")["atr"]
            parts.append(g.join(atr, how="inner"))
        sc = pd.concat(parts).dropna(subset=["atr"]).sort_index(kind="stable")
        bars = {s: store.load(store.klines_path(d, s, st.timeframe)) for s in st.symbols}
        fund = {s: store.load(store.funding_path(d, s)) for s in st.symbols}
        intra = {s: store.load(store.klines_path(d, s, "1m")) for s in st.symbols}
        intra = {k: v for k, v in intra.items() if v is not None}
        eq0 = st.backtest.initial_equity
        if equity:
            st.backtest.initial_equity = equity
        bp = params_from(st, 0.10, exit_score, rules)
        st.backtest.initial_equity = eq0
        bp.sl_atr, bp.tp_atr, bp.horizon = sl, 1000.0, self.cfg["max_hold"]
        bp.stop_order = "stop_limit"
        r = run_backtest(bars, sc[["symbol", "score", "atr"]], bp, fund, intra)
        return r, metrics.summary(r, bp.initial_equity)

    def walk_forward(self) -> WalkForwardScores:
        fc = self.forecasts()
        cfg = self.cfg
        rows, best = [], None
        for kind, tgt, share in itertools.product(cfg["model"], TARGETS, cfg["top_share"]):
            scores, q = self._score_frame(fc, f"{kind}_{tgt}", share)
            ins = scores[scores.index < self.cal_end]
            for sl, ex in itertools.product(cfg["sl_atr"], cfg["exit_score"]):
                _, m = self.backtest(ins, sl, ex)
                row = {"model": kind, "target": tgt, "top_share": share, "sl_atr": sl,
                       "exit_score": ex, **{k: m.get(k) for k in (
                           "trades", "total_return_pct", "max_drawdown_pct", "sharpe",
                           "profit_factor")}}
                rows.append(row)
                log.info("%s in-sample: %s", self.name, row)
                if (m["trades"] or 0) >= cfg["min_in_sample_trades"] and (
                        best is None or m["sharpe"] > best[0]["sharpe"]):
                    best = (row, scores, q)
        table = pd.DataFrame(rows)
        row, scores, q = best
        self.chosen = {k: row[k] for k in ("model", "target", "top_share", "sl_atr",
                                           "exit_score")} | {"q": q}
        oos = scores[scores.index >= self.cal_end]
        summary = (f"in-sample ({fc.index[0].date()} .. {self.cal_end.date()}) grid:\n"
                   f"{table.round(3).to_string()}\nchosen: {self.chosen}")
        return WalkForwardScores(oos, {"in_sample_grid": table}, summary)

    # --- live -----------------------------------------------------------------------------------
    def fit_final(self) -> dict:
        cfg, chosen = self.cfg, self.chosen
        if not chosen:
            raise RuntimeError(f"{self.name}: run the walk-forward first")
        syms = self._symbols()
        data = [self._bars(s) for s in syms]
        last = max(d.time[-1] for d in data)
        lo = last - pd.DateOffset(months=cfg["train_months"])
        rows = [np.flatnonzero(d.ok & np.isfinite(d.y[:, 0]) & (d.time >= lo)) for d in data]
        model = SlopeModel(cfg, chosen["model"]).fit(data, rows,
                                                     ((cfg["horizon"] + 1) * self.bar).value)
        version = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        folder = MODELS_DIR / self.name / version
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(model.state(), folder / "model.pt")
        meta = {"name": self.name, "version": version, "timeframe": self.settings.timeframe,
                "trained_until": str(last), **chosen}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2, default=float),
                                          encoding="utf-8")
        (MODELS_DIR / self.name / "current.json").write_text(json.dumps({"version": version}),
                                                            encoding="utf-8")
        return meta

    def load(self) -> None:
        root = MODELS_DIR / self.name
        version = json.loads((root / "current.json").read_text())["version"]
        meta = json.loads((root / version / "meta.json").read_text())
        if meta["timeframe"] != self.settings.timeframe:
            raise RuntimeError(f"{self.name}: trained on {meta['timeframe']}; retrain")
        self.chosen = meta
        self.model = SlopeModel.from_state(torch.load(root / version / "model.pt",
                                                      weights_only=False))

    def score(self, symbol: str, bar: pd.Timestamp) -> float | None:
        d = self._bars(symbol, live=True)
        pos = d.time.get_indexer([bar])[0]
        if pos < 0 or not d.ok[pos]:
            return None
        j = TARGETS.index(self.chosen["target"])
        raw = float(self.model.predict(d, np.array([pos]))[0, j])
        score = clip_score(raw / self.chosen["q"][symbol] * 0.10)
        log.info("%s %s bar %s | slope %+.3f | score %+.3f", self.name, symbol, bar, raw, score)
        return score
