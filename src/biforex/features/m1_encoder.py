"""A neural network pre-trained on 1-minute data; its forecasts become features for LightGBM.

The network reads the last 24 hours as 288 five-minute steps (move, range, close position,
volume, taker imbalance, trades, gaps) and learns to forecast what follows: realized
volatility, net move and extreme excursions over the next 4 and 24 hours, and volume. These
targets need no trade labels, and there is one training sample every 30 minutes (8 times more
than 4h bars), which is where a network can learn something LightGBM on 4h bars cannot see.

Everything is scale-free: moves are divided by the window's own volatility, volumes by the
window's mean, so BTC and ETH share one network.

Leakage control (see `research_features`): a forecast for a bar always comes from a network
that never saw that bar's future. Before `calibration_end` the bars are split into blocks and
each block is forecast by a network trained on the other blocks (cross-fitting, with gaps);
after it, a network retrained every `refit` (yearly) on all earlier data forecasts the next
period.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch import nn

log = logging.getLogger(__name__)

STEP = pd.Timedelta("5min")
WINDOW = 288  # 24 hours of 5-minute steps
STRIDE = 6  # one training sample every 30 minutes
H_SHORT, H_LONG = 48, 288  # forecast horizons: 4 and 24 hours
N_CHANNELS = 7
TARGETS = ["ret4h", "vol4h", "up4h", "down4h", "ret24h", "vol24h", "volume4h"]
FEATURES = [f"m1nn_{t}" for t in TARGETS]

DEFAULTS = {
    "seeds": 1, "lr": 2e-3, "weight_decay": 1e-3, "batch": 512, "max_epochs": 25,
    "patience": 4, "val_fraction": 0.15, "dropout": 0.1,
    "crossfit_blocks": 4, "calibration_end": "2023-04-01", "refit": "YS",  # yearly
}


# --- data -----------------------------------------------------------------------------------
def five_minute_steps(m1: pd.DataFrame) -> pd.DataFrame:
    """1m klines -> per-step channels on a complete 5-minute grid (gaps flagged)."""
    d = m1.set_index("open_time").sort_index()
    g = d.groupby(d.index.floor(STEP)).agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), buy=("taker_buy_volume", "sum"), trades=("trades", "sum"))
    grid = pd.date_range(g.index[0], g.index[-1], freq=STEP)
    g = g.reindex(grid)
    missing = g["close"].isna()
    close = g["close"].ffill()
    r = np.log(close).diff().fillna(0.0)
    rng = np.log(g["high"] / g["low"]).fillna(0.0)
    pos = ((g["close"] - g["low"]) / (g["high"] - g["low"])).fillna(0.5) - 0.5
    vol = g["volume"].fillna(0.0)
    imb = (2 * g["buy"] / vol.where(vol > 0) - 1).fillna(0.0)
    return pd.DataFrame({
        "r": r, "range": rng, "pos": pos, "lv": np.log1p(vol),
        "imb": imb, "lt": np.log1p(g["trades"].fillna(0.0)), "miss": missing.astype(float),
    }, index=grid).astype("float32")


class Steps:
    """Arrays of one symbol's 5-minute steps with fast window access."""

    def __init__(self, steps: pd.DataFrame):
        self.index = steps.index
        self.raw = steps.to_numpy(dtype=np.float32)
        r = self.raw[:, 0].astype(np.float64)
        self.c1 = np.concatenate([[0.0], np.cumsum(r)])  # cumulative move
        self.c2 = np.concatenate([[0.0], np.cumsum(r * r)])
        self.cv = np.concatenate([[0.0], np.cumsum(self.raw[:, 3].astype(np.float64))])

    def window_sd(self, end: np.ndarray) -> np.ndarray:
        """Std of 5-minute moves in the window ending at step `end` (inclusive)."""
        a, b = end + 1 - WINDOW, end + 1
        m = (self.c1[b] - self.c1[a]) / WINDOW
        v = (self.c2[b] - self.c2[a]) / WINDOW - m * m
        return np.sqrt(np.maximum(v, 1e-12)) + 1e-5

    def inputs(self, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(n, channels, WINDOW) normalized windows and (n, 1) log window volatility."""
        idx = end[:, None] + np.arange(1 - WINDOW, 1)[None, :]
        w = self.raw[idx]  # (n, WINDOW, 7)
        sd = self.window_sd(end)[:, None]
        x = np.empty_like(w)
        x[..., 0] = np.clip(w[..., 0] / sd, -10, 10)
        x[..., 1] = np.clip(w[..., 1] / sd, 0, 20)
        x[..., 2] = w[..., 2]
        x[..., 3] = w[..., 3] - w[..., 3].mean(axis=1, keepdims=True)
        x[..., 4] = w[..., 4]
        x[..., 5] = w[..., 5] - w[..., 5].mean(axis=1, keepdims=True)
        x[..., 6] = w[..., 6]
        return x.transpose(0, 2, 1).copy(), np.log(sd).astype(np.float32)

    def targets(self, end: np.ndarray) -> np.ndarray:
        """What follows step `end`, in units of the window's volatility (n, len(TARGETS))."""
        sd = self.window_sd(end)
        out = np.empty((len(end), len(TARGETS)), dtype=np.float64)
        for j, h in ((0, H_SHORT), (4, H_LONG)):
            a, b = end + 1, end + 1 + h
            move = self.c1[b] - self.c1[a]
            m = move / h
            v = np.maximum((self.c2[b] - self.c2[a]) / h - m * m, 1e-12)
            out[:, j] = np.clip(move / (sd * np.sqrt(h)), -6, 6)
            out[:, j + 1] = np.log(np.sqrt(v) / sd)
        path = self.c1[end[:, None] + 1 + np.arange(1, H_SHORT + 1)[None, :]] - self.c1[end + 1][:, None]
        scale = sd * np.sqrt(H_SHORT)
        out[:, 2] = np.clip(path.max(axis=1).clip(min=0) / scale, 0, 6)
        out[:, 3] = np.clip(-path.min(axis=1).clip(max=0) / scale, 0, 6)
        mean_lv_win = (self.cv[end + 1] - self.cv[end + 1 - WINDOW]) / WINDOW
        mean_lv_fut = (self.cv[end + 1 + H_SHORT] - self.cv[end + 1]) / H_SHORT
        out[:, 6] = mean_lv_fut - mean_lv_win
        return out.astype(np.float32)

    def valid_ends(self) -> np.ndarray:
        """Steps that have a full input window (forecast targets need H_LONG more)."""
        return np.arange(WINDOW - 1, len(self.index))

    def end_step(self, times: pd.DatetimeIndex) -> np.ndarray:
        """Index of the step ending at each time (the 5-minute step open at time - 5min);
        -1 if not available."""
        pos = self.index.get_indexer(times - STEP)
        pos[pos < WINDOW - 1] = -1
        return pos


# --- network --------------------------------------------------------------------------------
class EncoderNet(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()

        def block(i, o, k, s):
            return [nn.Conv1d(i, o, k, stride=s, padding=k // 2), nn.GELU(), nn.Dropout(dropout)]

        self.conv = nn.Sequential(
            *block(N_CHANNELS, 32, 7, 1), *block(32, 48, 5, 2), *block(48, 64, 5, 2),
            *block(64, 64, 5, 2), *block(64, 64, 3, 2))  # 288 -> 18 steps
        self.latent = nn.Sequential(nn.Linear(2 * 64 + 1, 32), nn.GELU())
        self.head = nn.Linear(32, len(TARGETS))

    def forward(self, x, log_sd):
        h = self.conv(x)
        z = self.latent(torch.cat([h.mean(-1), h[..., -1], log_sd], dim=1))
        return self.head(z)


class Encoder:
    """A few networks (seeds) trained on the same samples; forecasts are averaged."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.nets: list[EncoderNet] = []
        self.t_mean = self.t_std = None

    def fit(self, samples: list[tuple[Steps, np.ndarray]]) -> Encoder:
        """samples: (steps, end indices) per symbol; the latest `val_fraction` (by time)
        drive early stopping, after a gap of H_LONG steps. Input windows are built per batch."""
        cfg = self.cfg
        sts = [st for st, _ in samples]
        sid = np.concatenate([np.full(len(e), k) for k, (_, e) in enumerate(samples)])
        end = np.concatenate([e for _, e in samples]).astype(np.int64)
        Y = np.concatenate([st.targets(e) for st, e in samples])
        T = np.concatenate([st.index[e].as_unit("ns").asi8 for st, e in samples])
        cut = np.quantile(T, 1 - cfg["val_fraction"])
        gap = (H_LONG * STEP).value
        fit_idx, val_idx = np.flatnonzero(T < cut - gap), np.flatnonzero(T >= cut)
        self.t_mean, self.t_std = Y[fit_idx].mean(0), Y[fit_idx].std(0) + 1e-6
        yt = torch.from_numpy(((Y - self.t_mean) / self.t_std).astype(np.float32))

        def batch(b):
            x, ls = _gather(sts, sid[b], end[b])
            return torch.from_numpy(x), torch.from_numpy(ls)

        val_batches = [(batch(b), yt[b]) for b in
                       np.array_split(val_idx, max(1, len(val_idx) // 4096))]
        loss_fn = nn.HuberLoss()
        self.nets = []
        for seed in range(cfg["seeds"]):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            net = EncoderNet(cfg["dropout"])
            opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"],
                                    weight_decay=cfg["weight_decay"])
            best, best_state, bad = np.inf, None, 0
            for epoch in range(cfg["max_epochs"]):
                net.train()
                for b in np.array_split(rng.permutation(fit_idx),
                                        max(1, len(fit_idx) // cfg["batch"])):
                    x, ls = batch(b)
                    opt.zero_grad()
                    loss = loss_fn(net(x, ls), yt[b])
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    v = float(np.mean([loss_fn(net(*xb), yb).item() for xb, yb in val_batches]))
                if v < best - 1e-4:
                    best, bad = v, 0
                    best_state = {k: t.clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= cfg["patience"]:
                        break
            net.load_state_dict(best_state)
            net.eval()
            self.nets.append(net)
            log.info("encoder seed %d: %d samples, %d epochs, validation loss %.4f", seed,
                     len(fit_idx), epoch + 1, best)
        p = np.concatenate([self._predict(*xb) for xb, _ in val_batches])
        y = Y[val_idx]
        self.val_corr = {t: float(np.corrcoef(p[:, j], y[:, j])[0, 1])
                         for j, t in enumerate(TARGETS)}
        return self

    def _predict(self, x: torch.Tensor, ls: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            p = np.mean([n(x, ls).numpy() for n in self.nets], axis=0)
        return p * self.t_std + self.t_mean  # in target units

    def forecast(self, st: Steps, ends: np.ndarray) -> np.ndarray:
        out = []
        for b in np.array_split(np.asarray(ends), max(1, len(ends) // 4096)):
            x, ls = st.inputs(b)
            out.append(self._predict(torch.from_numpy(x), torch.from_numpy(ls)))
        return np.concatenate(out).astype(np.float32)

    def state(self) -> dict:
        return {"nets": [n.state_dict() for n in self.nets], "t_mean": self.t_mean,
                "t_std": self.t_std, "cfg": self.cfg}

    @classmethod
    def from_state(cls, blob: dict) -> Encoder:
        enc = cls(blob["cfg"])
        enc.t_mean, enc.t_std = blob["t_mean"], blob["t_std"]
        for s in blob["nets"]:
            net = EncoderNet(blob["cfg"]["dropout"])
            net.load_state_dict(s)
            net.eval()
            enc.nets.append(net)
        return enc


def _gather(sts: list[Steps], sid: np.ndarray, end: np.ndarray):
    x = np.empty((len(end), N_CHANNELS, WINDOW), dtype=np.float32)
    ls = np.empty((len(end), 1), dtype=np.float32)
    for k in np.unique(sid):
        m = sid == k
        x[m], ls[m] = sts[k].inputs(end[m])
    return x, ls


# --- research features without leakage ------------------------------------------------------
def _training_ends(st: Steps, lo: pd.Timestamp | None, hi: pd.Timestamp) -> np.ndarray:
    """Sample steps whose window and 24h forecast span lie inside [lo, hi)."""
    ends = st.valid_ends()[::STRIDE]
    ends = ends[ends + H_LONG < len(st.index)]
    start = st.index[ends - (WINDOW - 1)]
    stop = st.index[ends + H_LONG] + STEP
    ok = stop <= hi
    if lo is not None:
        ok &= start >= lo
    return ends[ok]


def research_features(steps: dict[str, Steps], bar_close: dict[str, pd.DatetimeIndex],
                      cfg: dict) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Out-of-sample forecasts at every bar close. bar_close: symbol -> close times of its
    bars. Returns symbol -> (n_bars, len(TARGETS)) array (NaN where not available) and a
    table of the trained networks with their validation correlations."""
    cal_end = pd.Timestamp(cfg["calibration_end"], tz="UTC")
    first = min(s.index[WINDOW] for s in steps.values())
    last = max(s.index[-1] for s in steps.values()) + STEP
    out = {s: np.full((len(t), len(TARGETS)), np.nan, dtype=np.float32)
           for s, t in bar_close.items()}
    ends = {s: steps[s].end_step(t) for s, t in bar_close.items()}
    gap = (WINDOW + H_LONG) * STEP
    rows = []

    def apply(enc: Encoder, lo: pd.Timestamp, hi: pd.Timestamp, how: str) -> None:
        for s, t in bar_close.items():
            m = (t >= lo) & (t < hi) & (ends[s] >= 0)
            if m.any():
                out[s][m] = enc.forecast(steps[s], ends[s][m])
        rows.append({"how": how, "applies_from": lo, "applies_to": hi,
                     **{f"corr_{k}": round(v, 3) for k, v in enc.val_corr.items()}})
        log.info("encoder %s -> %s..%s: validation corr %s", how, lo.date(), hi.date(),
                 {k: round(v, 3) for k, v in enc.val_corr.items()})

    # before calibration_end: cross-fitting over contiguous blocks
    edges = pd.date_range(first, cal_end, periods=cfg["crossfit_blocks"] + 1)
    for k in range(cfg["crossfit_blocks"]):
        lo, hi = edges[k], edges[k + 1]
        samples = []
        for st in steps.values():
            before = _training_ends(st, None, lo - gap) if k > 0 else np.array([], int)
            after = _training_ends(st, hi + gap, cal_end)
            samples.append((st, np.concatenate([before, after])))
        enc = Encoder(cfg).fit(samples)
        apply(enc, lo if k > 0 else bar_close_min(bar_close), hi, f"crossfit block {k + 1}")
    # after it: retrained every `refit` period on all earlier data
    starts = [cal_end, *pd.date_range(cal_end, last, freq=cfg["refit"], inclusive="neither")]
    for lo, hi in zip(starts, [*starts[1:], last + pd.Timedelta("1D")], strict=True):
        enc = Encoder(cfg).fit([(st, _training_ends(st, None, lo)) for st in steps.values()])
        apply(enc, lo, hi, f"trained before {lo.date()}")
    return out, pd.DataFrame(rows)


def bar_close_min(bar_close: dict[str, pd.DatetimeIndex]) -> pd.Timestamp:
    return min(t.min() for t in bar_close.values())
