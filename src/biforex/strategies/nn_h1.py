"""Neural networks trained on trading profit for the 1-hour timeframe (nn1_* strategies).

Three variants share this module:
  nn1_mlp        V1: MLP on the feature snapshot + changes of key features over 1/3/6 bars
  nn1_hybrid     V2: two branches - a 1D-CNN over the last `window` raw bars and an MLP on
                 the feature snapshot - joined before the output
  nn1_hybrid_es  V3: V2, then the output layer is fine-tuned by an evolution strategy on the
                 EXACT trading rule (threshold, one position per symbol, >= N trades a day,
                 costs), which gradients cannot see
All output p = tanh(...) in [-1, +1] and are trained by gradient ascent on
    mean(pnl) - risk_aversion * std(pnl),  pnl = max(p,0)*long_net + max(-p,0)*short_net
where long_net / short_net are triple-barrier trade results (take profit tp_atr, stop sl_atr,
time limit `horizon` bars) minus costs, computed per strategy (the horizon is a parameter).
Training uses all `data.extra_symbols` too; only config `symbols` are traded.

Trade frequency: the entry threshold is set on the validation tail of each training window
so that the traded symbols together make >= min_trades_per_day non-overlapping trades a day
there (the "hard" requirement). A second, unconstrained calibration (top `free_top_share` of
bars) is saved as oos_scores_free.parquet for comparison.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..config import PROJECT_ROOT
from ..data import store
from ..data.schema import timeframe_delta
from ..labels.triple_barrier import triple_barrier
from ..model.registry import MODELS_DIR
from ..model.walkforward import monthly_folds
from . import register
from .base import Strategy, WalkForwardScores, clip_score
from .lgbm_tb import LgbmTripleBarrier
from .nn_seq import step_features

log = logging.getLogger(__name__)
torch.set_num_threads(6)

DEFAULTS = {
    "horizon": 12, "tp_atr": 2.0, "sl_atr": 1.0,
    "hidden": [32, 16], "dropout": 0.2, "weight_decay": 1e-3, "lr": 1e-3,
    "batch": 8192, "max_epochs": 40, "patience": 6,
    "risk_aversion": 0.5, "cost": 0.001, "seeds": 2,
    "val_fraction": 0.2, "retrain_every": 3,
    "min_trades_per_day": 2.0, "fill_rate": 0.85, "free_top_share": 0.06,
    "window": 32, "channels": 16,
    "es": False, "es_iterations": 40, "es_population": 24, "es_sigma": 0.05, "es_lr": 0.05,
}
DIFF_FEATURES = ["ret_1", "ret_8", "rsi14", "macd_hist_atr", "ema21_dist_atr", "ema55_dist_atr",
                 "bb_pos", "vol_rel24", "taker_buy_ratio", "taker_imbalance24", "rv24",
                 "atr_pct", "range_pos24", "oi_chg_4", "global_ls_log", "top_ls_positions_log",
                 "funding_last", "stoch_k", "cci20", "dist_high24_atr"]
DIFF_LAGS = (1, 3, 6)


# --- data ---------------------------------------------------------------------------------
class SymbolData:
    """Per-symbol arrays aligned by row: snapshot inputs, raw steps, outcomes."""

    def __init__(self, snap: np.ndarray, steps: np.ndarray, times: pd.DatetimeIndex,
                 long_net: np.ndarray, short_net: np.ndarray, long_bars: np.ndarray,
                 short_bars: np.ndarray, rows: np.ndarray):
        self.snap, self.steps, self.times = snap, steps, times
        self.long_net, self.short_net = long_net, short_net
        self.long_bars, self.short_bars = long_bars, short_bars
        self.rows = rows  # positions of usable rows (labelled) in snap/steps


def snapshot_inputs(f: pd.DataFrame) -> pd.DataFrame:
    parts = [f]
    for lag in DIFF_LAGS:
        cols = [c for c in DIFF_FEATURES if c in f]
        parts.append((f[cols] - f[cols].shift(lag)).add_suffix(f"_d{lag}"))
    return pd.concat(parts, axis=1).astype("float32")


class Scaler:
    def fit(self, X: np.ndarray) -> Scaler:
        self.med = np.nanmedian(X, axis=0)
        q75, q25 = np.nanpercentile(X, 75, axis=0), np.nanpercentile(X, 25, axis=0)
        self.iqr = np.where((q75 - q25) > 0, q75 - q25, 1.0)
        self.med = np.nan_to_num(self.med)
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.clip(np.nan_to_num((X - self.med) / self.iqr), -5, 5).astype(np.float32)


# --- networks -------------------------------------------------------------------------------
class MlpNet(nn.Module):
    def __init__(self, n_snap: int, hidden: list[int], dropout: float, **_):
        super().__init__()
        layers, prev = [], n_snap
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def features(self, snap, seq=None):
        return self.body(snap)

    def forward(self, snap, seq=None):
        return torch.tanh(self.head(self.features(snap, seq))).squeeze(-1)


class HybridNet(nn.Module):
    def __init__(self, n_snap: int, n_step: int, hidden: list[int], dropout: float,
                 channels: int, **_):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_step, channels, 3, padding=1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=4, dilation=4), nn.GELU())
        self.snap = nn.Sequential(nn.Linear(n_snap, hidden[0]), nn.GELU(), nn.Dropout(dropout))
        self.join = nn.Sequential(nn.Linear(2 * channels + hidden[0], hidden[-1]), nn.GELU(),
                                  nn.Dropout(dropout))
        self.head = nn.Linear(hidden[-1], 1)

    def features(self, snap, seq):
        h = self.conv(seq.transpose(1, 2))
        z = torch.cat([h.mean(-1), h[..., -1], self.snap(snap)], dim=1)
        return self.join(z)

    def forward(self, snap, seq):
        return torch.tanh(self.head(self.features(snap, seq))).squeeze(-1)


def objective(p, ln, sn, risk_aversion):
    pnl = torch.relu(p) * ln + torch.relu(-p) * sn
    return pnl.mean() - risk_aversion * pnl.std()


# --- trade simulation (for frequency calibration and ES fitness) ----------------------------
def simulate(p: np.ndarray, th: float, groups: list[np.ndarray], times: np.ndarray,
             ln: np.ndarray, sn: np.ndarray, lb: np.ndarray, sb: np.ndarray,
             bar_ns: int) -> tuple[np.ndarray, float]:
    """Non-overlapping trades per symbol (group = row positions of one symbol, time-sorted).
    Returns trade pnls and days covered."""
    pnls = []
    t_min, t_max = np.inf, -np.inf
    for g in groups:
        if len(g) == 0:
            continue
        t_min, t_max = min(t_min, times[g[0]]), max(t_max, times[g[-1]])
        sig = g[np.abs(p[g]) >= th]
        busy = -np.inf
        for i in sig:
            if times[i] < busy:
                continue
            if p[i] > 0:
                pnls.append(ln[i])
                busy = times[i] + (lb[i] + 1) * bar_ns
            else:
                pnls.append(sn[i])
                busy = times[i] + (sb[i] + 1) * bar_ns
    days = max((t_max - t_min) / 86_400e9, 1.0) if t_max > t_min else 1.0
    return np.asarray(pnls), days


def sharpe(pnls: np.ndarray) -> float:
    return float(pnls.mean() / (pnls.std() + 1e-9) * np.sqrt(len(pnls))) if len(pnls) > 1 else -1


class ProfitModel:
    """Seeds of one network type + scalers + two calibrated thresholds, one training window."""

    def __init__(self, cfg: dict, mode: str, bar: pd.Timedelta):
        self.cfg, self.mode, self.bar_ns = cfg, mode, bar.value

    def _batch_inputs(self, data: dict[str, SymbolData], idx: np.ndarray, sym: np.ndarray):
        snap = np.empty((len(idx), self.n_snap), dtype=np.float32)
        seq = (np.empty((len(idx), self.cfg["window"], self.n_step), dtype=np.float32)
               if self.mode == "hybrid" else None)
        for s, d in data.items():
            m = sym == s
            if not m.any():
                continue
            r = idx[m]
            snap[m] = self.snap_scaler(d.snap[r])
            if seq is not None:
                w = self.cfg["window"]
                pos = r[:, None] + np.arange(-w + 1, 1)[None, :]
                raw = d.steps[np.clip(pos, 0, None)]
                raw[pos < 0] = 0.0
                seq[m] = self.step_scaler(raw.reshape(-1, self.n_step)).reshape(raw.shape)
        return torch.from_numpy(snap), None if seq is None else torch.from_numpy(seq)

    def predict(self, data, idx, sym, nets=None, chunk=50_000) -> np.ndarray:
        nets = nets or self.nets
        out = []
        with torch.no_grad():
            for a in range(0, len(idx), chunk):
                snap, seq = self._batch_inputs(data, idx[a:a + chunk], sym[a:a + chunk])
                out.append(np.mean([n(snap, seq).numpy() for n in nets], axis=0))
        return np.concatenate(out) if out else np.empty(0)

    def fit(self, data: dict[str, SymbolData], idx: np.ndarray, sym: np.ndarray,
            times: np.ndarray, traded: list[str]) -> ProfitModel:
        cfg = self.cfg
        order = np.argsort(times, kind="stable")
        idx, sym, times = idx[order], sym[order], times[order]
        purge = (cfg["horizon"] + 1) * self.bar_ns
        val_start = times[int(len(times) * (1 - cfg["val_fraction"]))]
        fit_m, val_m = times < val_start - purge, times >= val_start
        any_d = next(iter(data.values()))
        self.n_snap, self.n_step = any_d.snap.shape[1], any_d.steps.shape[1]
        snap_rows = np.concatenate([data[s].snap[idx[fit_m & (sym == s)]] for s in data])
        self.snap_scaler = Scaler().fit(snap_rows)
        step_rows = np.concatenate([data[s].steps[idx[fit_m & (sym == s)]] for s in data])
        self.step_scaler = Scaler().fit(step_rows)
        ln = np.empty(len(idx), np.float32)
        sn = np.empty(len(idx), np.float32)
        lb = np.empty(len(idx), np.int64)
        sb = np.empty(len(idx), np.int64)
        for s, d in data.items():
            m = sym == s
            ln[m], sn[m] = d.long_net[idx[m]], d.short_net[idx[m]]
            lb[m], sb[m] = d.long_bars[idx[m]], d.short_bars[idx[m]]
        Net = HybridNet if self.mode == "hybrid" else MlpNet
        self.nets = []
        fit_idx = np.flatnonzero(fit_m)
        val_snap, val_seq = self._batch_inputs(data, idx[val_m], sym[val_m])
        lnv, snv = torch.from_numpy(ln[val_m]), torch.from_numpy(sn[val_m])
        for seed in range(cfg["seeds"]):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            net = Net(n_snap=self.n_snap, n_step=self.n_step, hidden=cfg["hidden"],
                      dropout=cfg["dropout"], channels=cfg["channels"])
            opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"],
                                    weight_decay=cfg["weight_decay"])
            best, state, bad = -np.inf, None, 0
            for _ in range(cfg["max_epochs"]):
                net.train()
                for b in np.array_split(rng.permutation(fit_idx),
                                        max(1, len(fit_idx) // cfg["batch"])):
                    snap, seq = self._batch_inputs(data, idx[b], sym[b])
                    opt.zero_grad()
                    loss = -objective(net(snap, seq), torch.from_numpy(ln[b]),
                                      torch.from_numpy(sn[b]), cfg["risk_aversion"])
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    v = objective(net(val_snap, val_seq), lnv, snv, cfg["risk_aversion"]).item()
                if v > best:
                    best, bad = v, 0
                    state = {k: t.clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= cfg["patience"]:
                        break
            net.load_state_dict(state)
            net.eval()
            self.nets.append(net)
        self.val_objective = float(best)

        # thresholds on the validation tail, traded symbols only
        p_val = self.predict(data, idx[val_m], sym[val_m])
        tv, sv = times[val_m], sym[val_m]
        groups = [np.flatnonzero(sv == s) for s in traded]
        vl, vs, vlb, vsb = ln[val_m], sn[val_m], lb[val_m], sb[val_m]
        if cfg["es"]:
            self._evolve_head(data, idx, sym, fit_m, ln, sn, lb, sb, times, traded)
            p_val = self.predict(data, idx[val_m], sym[val_m])
        traded_abs = np.abs(p_val[np.isin(sv, traded)])
        need = cfg["min_trades_per_day"] / cfg["fill_rate"]
        self.th_hard = float(np.quantile(traded_abs, 0.5))
        for q in np.linspace(0.995, 0.5, 100):
            th = float(np.quantile(traded_abs, q))
            pn, days = simulate(p_val, th, groups, tv.astype(np.int64), vl, vs, vlb, vsb,
                                self.bar_ns)
            if len(pn) / days >= need:
                self.th_hard = th
                self.val_trades_day = len(pn) / days
                self.val_sharpe = sharpe(pn)
                break
        self.th_free = float(np.quantile(traded_abs, 1 - cfg["free_top_share"])) or 1.0
        return self

    def _evolve_head(self, data, idx, sym, fit_m, ln, sn, lb, sb, times, traded):
        """ES on the output layer: fitness = Sharpe of the exact thresholded trades on the
        fitting part, x penalty when trades a day fall below the requirement."""
        cfg = self.cfg
        m = fit_m & np.isin(sym, traded)
        fi = np.flatnonzero(m)
        snap, seq = self._batch_inputs(data, idx[fi], sym[fi])
        with torch.no_grad():
            hs = [n.features(snap, seq).numpy() for n in self.nets]
        t = times[fi].astype(np.int64)
        groups = [np.flatnonzero(sym[fi] == s) for s in traded]
        need = cfg["min_trades_per_day"] / cfg["fill_rate"]

        def fitness(heads):
            p = np.mean([np.tanh(h @ w[:-1] + w[-1]) for h, w in zip(hs, heads, strict=True)],
                        axis=0)
            th = float(np.quantile(np.abs(p), 0.9))
            pn, days = simulate(p, th, groups, t, ln[fi], sn[fi], lb[fi], sb[fi], self.bar_ns)
            return sharpe(pn) * min(1.0, (len(pn) / days) / need)

        heads = [np.concatenate([n.head.weight.detach().numpy()[0],
                                 n.head.bias.detach().numpy()]) for n in self.nets]
        rng = np.random.default_rng(0)
        best, best_heads = fitness(heads), [h.copy() for h in heads]
        for _ in range(cfg["es_iterations"]):
            eps = [rng.normal(0, 1, (cfg["es_population"], len(h))) for h in heads]
            pairs = list(zip(heads, eps, strict=True))
            f_pos = np.array([fitness([h + cfg["es_sigma"] * e[k] for h, e in pairs])
                              for k in range(cfg["es_population"])])
            f_neg = np.array([fitness([h - cfg["es_sigma"] * e[k] for h, e in pairs])
                              for k in range(cfg["es_population"])])
            adv = (f_pos - f_neg) / ((f_pos - f_neg).std() + 1e-9)
            heads = [h + cfg["es_lr"] / (2 * cfg["es_sigma"]) * (adv[:, None] * e).mean(0)
                     for h, e in zip(heads, eps, strict=True)]
            f = fitness(heads)
            if f > best:
                best, best_heads = f, [h.copy() for h in heads]
        for n, h in zip(self.nets, best_heads, strict=True):
            with torch.no_grad():
                n.head.weight[0] = torch.from_numpy(h[:-1].astype(np.float32))
                n.head.bias[0] = float(h[-1])
        self.es_fitness = best


class NnH1Base(Strategy):
    mode = "mlp"
    use_es = False

    def __init__(self, settings, params: dict | None = None):
        super().__init__(settings, params)
        self.cfg = {**DEFAULTS, **self.params, "es": self.use_es or self.params.get("es", False)}
        self.name = f"{type(self).name}_h{self.cfg['horizon']}"
        self.features_src = LgbmTripleBarrier(settings)
        self.bar = timeframe_delta(settings.timeframe)

    def _outcomes(self, symbol: str) -> pd.DataFrame:
        st, cfg = self.settings, self.cfg
        path = (PROJECT_ROOT / "data" / "strategies" / "nn_h1" / st.timeframe
                / f"h{cfg['horizon']}_tp{cfg['tp_atr']}_sl{cfg['sl_atr']}" / f"{symbol}.parquet")
        if path.exists():
            return pd.read_parquet(path)
        k = store.load(store.klines_path(st.data.dir, symbol, st.timeframe))
        intra = store.load(store.klines_path(st.data.dir, symbol, "1m"))
        o = triple_barrier(k, cfg["horizon"], cfg["tp_atr"], cfg["sl_atr"], 14, intra, self.bar)
        path.parent.mkdir(parents=True, exist_ok=True)
        o.to_parquet(path)
        return o

    def _snapshot(self, symbol: str, live: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        f = self.features_src.features(symbol, live)
        snap = snapshot_inputs(f)
        snap["symbol_id"] = np.float32(self.settings.all_symbols.index(symbol)
                                       if symbol in self.settings.all_symbols else 0)
        return f, snap

    def _symbol_data(self, symbol: str, live: bool = False) -> SymbolData:
        f, snap = self._snapshot(symbol, live)
        snap = snap.reindex(columns=self.columns)  # same inputs for every symbol
        steps = step_features(f)
        n = len(f)
        if live:
            z = np.zeros(n, np.float32)
            return SymbolData(snap.to_numpy(), steps.to_numpy(), f.index, z, z, z.astype(int),
                              z.astype(int), np.arange(n))
        o = self._outcomes(symbol).reindex(f.index)
        ok = o["label"].notna().to_numpy()
        cost = self.cfg["cost"]
        return SymbolData(snap.to_numpy(), steps.to_numpy(), f.index,
                          (o["long_ret"] - cost).fillna(0).to_numpy(np.float32),
                          (o["short_ret"] - cost).fillna(0).to_numpy(np.float32),
                          o["long_bars"].fillna(0).to_numpy(np.int64),
                          o["short_bars"].fillna(0).to_numpy(np.int64), np.flatnonzero(ok))

    def _all(self):
        cols: set[str] = set()
        for s in self.settings.all_symbols:
            cols |= set(self._snapshot(s, live=False)[1].columns)
        self.columns = sorted(cols)
        data = {s: self._symbol_data(s) for s in self.settings.all_symbols}
        idx = np.concatenate([d.rows for d in data.values()])
        sym = np.concatenate([np.full(len(d.rows), s) for s, d in data.items()])
        times = np.concatenate([d.times[d.rows].as_unit("ns").asi8 for d in data.values()])
        order = np.argsort(times, kind="stable")
        return data, idx[order], sym[order], times[order]

    def walk_forward(self) -> WalkForwardScores:
        st = self.settings
        data, idx, sym, times = self._all()
        tindex = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
        purge = (self.cfg["horizon"] + 1) * self.bar
        traded = st.symbols
        hard, free, rows, model = [], [], [], None
        for k, fold in enumerate(monthly_folds(tindex, st.retrain.train_months, purge)):
            te = (tindex >= fold.test_start) & (tindex < fold.test_end) & np.isin(sym, traded)
            if te.sum() == 0:
                continue
            if model is None or k % self.cfg["retrain_every"] == 0:
                tr = (tindex >= fold.train_start) & (tindex < fold.train_end)
                model = ProfitModel(self.cfg, self.mode, self.bar).fit(
                    data, idx[tr], sym[tr], times[tr], traded)
                rows.append({"fold": k, "test_start": fold.test_start,
                             "val_objective": model.val_objective,
                             "val_trades_day": getattr(model, "val_trades_day", np.nan),
                             "val_sharpe": getattr(model, "val_sharpe", np.nan),
                             "th_hard": model.th_hard, "th_free": model.th_free})
                log.info("%s: fold %d trained, val objective %.5f, hard threshold %.3f "
                         "(%.2f trades/day on validation)", self.name, k, model.val_objective,
                         model.th_hard, getattr(model, "val_trades_day", np.nan))
            p = model.predict(data, idx[te], sym[te])
            base = pd.DataFrame({"symbol": sym[te]}, index=tindex[te])
            hard.append(base.assign(score=np.clip(p / model.th_hard * 0.10, -1, 1)))
            free.append(base.assign(score=np.clip(p / model.th_free * 0.10, -1, 1)))
        hard = pd.concat(hard).sort_index(kind="stable")
        free = pd.concat(free).sort_index(kind="stable")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        free.to_parquet(self.report_dir / "oos_scores_free.parquet")
        folds = pd.DataFrame(rows).set_index("fold")
        summary = (f"{len(folds)} trainings x {self.cfg['seeds']} seeds, horizon "
                   f"{self.cfg['horizon']} bars; median validation objective "
                   f"{folds['val_objective'].median():.5f}; validation trades/day at the hard "
                   f"threshold median {folds['val_trades_day'].median():.2f}\n"
                   f"{folds.round(4).to_string()}")
        return WalkForwardScores(hard, {"trainings": folds}, summary)

    def fit_final(self) -> dict:
        st = self.settings
        data, idx, sym, times = self._all()
        start = (pd.Timestamp(times[-1], tz="UTC")
                 - pd.DateOffset(months=st.retrain.train_months)).value
        m = times > start
        model = ProfitModel(self.cfg, self.mode, self.bar).fit(data, idx[m], sym[m], times[m],
                                                               st.symbols)
        version = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        folder = MODELS_DIR / self.name / version
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({"nets": [n.state_dict() for n in model.nets], "snap": model.snap_scaler,
                    "step": model.step_scaler, "th": model.th_hard, "n_snap": model.n_snap,
                    "n_step": model.n_step, "columns": self.columns}, folder / "model.pt")
        meta = {"name": self.name, "version": version, "timeframe": st.timeframe,
                "cfg": self.cfg, "mode": self.mode}
        (folder / "meta.json").write_text(json.dumps(meta, default=float), encoding="utf-8")
        (MODELS_DIR / self.name / "current.json").write_text(json.dumps({"version": version}),
                                                            encoding="utf-8")
        return meta

    def load(self) -> None:
        version = json.loads((MODELS_DIR / self.name / "current.json").read_text())["version"]
        blob = torch.load(MODELS_DIR / self.name / version / "model.pt", weights_only=False)
        model = ProfitModel(self.cfg, self.mode, self.bar)
        model.snap_scaler, model.step_scaler = blob["snap"], blob["step"]
        model.n_snap, model.n_step, model.th_hard = blob["n_snap"], blob["n_step"], blob["th"]
        self.columns = blob["columns"]
        Net = HybridNet if self.mode == "hybrid" else MlpNet
        model.nets = []
        for state in blob["nets"]:
            net = Net(n_snap=model.n_snap, n_step=model.n_step, hidden=self.cfg["hidden"],
                      dropout=self.cfg["dropout"], channels=self.cfg["channels"])
            net.load_state_dict(state)
            net.eval()
            model.nets.append(net)
        self.model = model

    def score(self, symbol: str, bar: pd.Timestamp) -> float | None:
        d = self._symbol_data(symbol, live=True)
        if bar not in d.times:
            return None
        i = np.array([d.times.get_loc(bar)])
        p = self.model.predict({symbol: d}, i, np.array([symbol]))[0]
        score = clip_score(p / self.model.th_hard * 0.10)
        log.info("%s %s bar %s | p %+.3f | score %+.3f", self.name, symbol, bar, p, score)
        return score


@register
class Nn1Mlp(NnH1Base):
    name = "nn1_mlp"
    mode = "mlp"


@register
class Nn1Hybrid(NnH1Base):
    name = "nn1_hybrid"
    mode = "hybrid"


@register
class Nn1HybridEs(NnH1Base):
    name = "nn1_hybrid_es"
    mode = "hybrid"
    use_es = True
