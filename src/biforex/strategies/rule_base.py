"""Common machinery for rule strategies (no ML): pick parameters on in-sample data only.

A subclass defines the parameter grid, the raw signal and how it becomes a score. For every
parameter set the rule is backtested on data before `calibration_end`; the best one by
Sharpe (with enough trades) is kept. Saved walk-forward scores start at `calibration_end`,
so they are out-of-sample for the parameter choice.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import ClassVar

import pandas as pd

from ..data import store
from ..labels.build import labels_path
from ..model.registry import MODELS_DIR
from .base import Strategy, WalkForwardScores, clip_score

log = logging.getLogger(__name__)

RULE_DEFAULTS = {
    "calibration_end": "2023-04-01",  # same start as lgbm_tb's out-of-sample period
    "selection_threshold": 0.10,
    "min_in_sample_trades": 30,
}


class RuleStrategy(Strategy):
    defaults: ClassVar[dict] = {}

    def __init__(self, settings, params: dict | None = None):
        super().__init__(settings, params)
        self.cfg = {**RULE_DEFAULTS, **self.defaults, **self.params}
        self.cal_end = pd.Timestamp(self.cfg["calibration_end"], tz="UTC")
        self.chosen: dict = {}  # params + fitted values (e.g. scales), saved / loaded

    # --- to implement ------------------------------------------------------------------
    def grid(self) -> list[dict]:
        raise NotImplementedError

    def raw(self, symbol: str, params: dict, live: bool = False) -> pd.Series:
        """Raw signal per bar (index open_time), NaN where not computable."""
        raise NotImplementedError

    def fit(self, raw_in_sample: pd.Series, params: dict) -> dict:
        """Values fitted on in-sample raw signal of one symbol (e.g. a scale)."""
        return {}

    def to_score(self, raw: pd.Series, params: dict, fitted: dict) -> pd.Series:
        raise NotImplementedError

    # --- shared --------------------------------------------------------------------------
    def _scores(self, params: dict) -> tuple[pd.DataFrame, dict]:
        parts, fitted = [], {}
        for s in self.settings.symbols:
            raw = self.raw(s, params).dropna()
            fitted[s] = self.fit(raw[raw.index < self.cal_end], params)
            score = self.to_score(raw, params, fitted[s]).clip(-1, 1)
            parts.append(pd.DataFrame({"symbol": s, "score": score}))
        return pd.concat(parts).sort_index(kind="stable"), fitted

    def _in_sample_backtest(self, scores: pd.DataFrame) -> dict:
        from ..backtest import metrics
        from ..backtest.engine import params_from, run_backtest

        st, d = self.settings, self.settings.data.dir
        sc = scores[scores.index < self.cal_end]
        parts = []
        for s, g in sc.groupby("symbol"):
            atr = store.load(labels_path(st, s)).set_index("open_time")["atr"]
            parts.append(g.join(atr, how="inner"))
        sc = pd.concat(parts).dropna(subset=["atr"]).sort_index(kind="stable")
        bars = {s: store.load(store.klines_path(d, s, st.timeframe)) for s in st.symbols}
        fund = {s: store.load(store.funding_path(d, s)) for s in st.symbols}
        th = self.cfg["selection_threshold"]
        r = run_backtest(bars, sc[["symbol", "score", "atr"]], params_from(st, th, th), fund)
        m = metrics.summary(r, st.backtest.initial_equity)
        return {k: m.get(k) for k in ("trades", "total_return_pct", "max_drawdown_pct",
                                      "sharpe", "profit_factor")}

    def _select(self) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
        rows, best, best_run = [], None, None
        for params in self.grid():
            scores, fitted = self._scores(params)
            m = self._in_sample_backtest(scores)
            rows.append({**params, **m})
            if (m["trades"] or 0) >= self.cfg["min_in_sample_trades"] and (
                    best is None or m["sharpe"] > best[1]["sharpe"]):
                best, best_run = (params, m), (scores, fitted)
        table = pd.DataFrame(rows)
        if best is None:
            raise RuntimeError(f"{self.name}: no parameter set reached "
                               f"{self.cfg['min_in_sample_trades']} in-sample trades\n{table}")
        return {**best[0], "fitted": best_run[1]}, best_run[0], table

    def walk_forward(self) -> WalkForwardScores:
        chosen, scores, table = self._select()
        self.chosen = chosen
        oos = scores[scores.index >= self.cal_end]
        summary = (f"in-sample (< {self.cal_end.date()}) backtests, threshold "
                   f"{self.cfg['selection_threshold']}:\n{table.round(3).to_string()}\n"
                   f"chosen: {chosen}\nsaved scores are out-of-sample: "
                   f"{oos.index[0].date()} -> {oos.index[-1].date()}")
        return WalkForwardScores(oos, {"in_sample_grid": table}, summary)

    def fit_final(self) -> dict:
        chosen = self.chosen or self._select()[0]
        version = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        folder = MODELS_DIR / self.name / version
        folder.mkdir(parents=True, exist_ok=True)
        meta = {"name": self.name, "version": version, "timeframe": self.settings.timeframe,
                "calibration_end": str(self.cal_end), **chosen}
        (folder / "params.json").write_text(json.dumps(meta, indent=2, default=float),
                                            encoding="utf-8")
        (MODELS_DIR / self.name / "current.json").write_text(json.dumps({"version": version}),
                                                            encoding="utf-8")
        log.info("%s: parameters saved to %s", self.name, folder)
        return meta

    def load(self) -> None:
        version = json.loads((MODELS_DIR / self.name / "current.json").read_text())["version"]
        meta = json.loads((MODELS_DIR / self.name / version / "params.json").read_text())
        if meta["timeframe"] != self.settings.timeframe:
            raise RuntimeError(f"{self.name}: calibrated on {meta['timeframe']}, config is "
                               f"{self.settings.timeframe}; rerun train.py")
        self.chosen = meta
        log.info("%s/%s: %s", self.name, version,
                 {k: v for k, v in meta.items() if k in self.grid()[0]})

    def score(self, symbol: str, bar: pd.Timestamp) -> float | None:
        params = {k: self.chosen[k] for k in self.grid()[0]}
        raw = self.raw(symbol, params, live=True)
        if bar not in raw.index or pd.isna(raw.at[bar]):
            return None
        s = self.to_score(raw.loc[[bar]], params, self.chosen["fitted"][symbol]).iloc[0]
        score = clip_score(s)
        log.info("%s %s bar %s | raw %+.3f | score %+.3f", self.name, symbol, bar,
                 raw.at[bar], score)
        return score
