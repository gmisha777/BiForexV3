"""Strategy lgbm_m1nn: lgbm_tb's features plus forecasts of a network pre-trained on 1m data.

The network (features/m1_encoder.py) reads the last 24 hours of 5-minute steps and forecasts
volatility, move, excursions and volume for the next 4 / 24 hours, without trade labels.
Its 7 forecasts at each bar close are added to the lgbm_tb feature table; LightGBM is then
trained on the shared triple-barrier labels exactly as lgbm_tb. Only symbols with stored 1m
data are used (BTCUSDT, ETHUSDT).

Research forecasts never come from a network that saw the bar's future (cross-fitting before
`calibration_end`, yearly retraining after it), and saved scores start at `calibration_end`.
Live: a network trained on all data by fit_final.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import torch

from ..config import PROJECT_ROOT, Settings
from ..data import store
from ..data.schema import timeframe_delta
from ..features import m1_encoder as m1
from ..model.registry import MODELS_DIR
from . import register
from .base import WalkForwardScores
from .lgbm_base import LgbmStrategy
from .lgbm_tb import LgbmTripleBarrier

log = logging.getLogger(__name__)

LIVE_DAYS = 3  # 1m history used live: one 24h window plus margin
ENCODER_KEYS = set(m1.DEFAULTS)


@register
class LgbmM1Net(LgbmStrategy):
    name = "lgbm_m1nn"
    data_intervals = ("1m",)

    def __init__(self, settings: Settings, params: dict | None = None):
        params = params or {}
        self.enc_cfg = {**m1.DEFAULTS, **{k: v for k, v in params.items() if k in ENCODER_KEYS}}
        super().__init__(settings, {k: v for k, v in params.items() if k not in ENCODER_KEYS})
        self.base = LgbmTripleBarrier(settings)
        self.encoder: m1.Encoder | None = None
        self._cache: dict[str, pd.DataFrame] = {}

    def base_model_config(self, settings: Settings) -> dict:
        return settings.model.model_dump()

    def _m1_path(self, symbol: str):
        return store.klines_path(self.st.data.dir, symbol, "1m")

    def _has_data(self, symbol: str) -> bool:
        return self._m1_path(symbol).exists()

    def _cache_path(self, symbol: str):
        return (PROJECT_ROOT / "data" / "strategies" / self.name / self.settings.report_key
                / f"{symbol}.parquet")

    # --- features --------------------------------------------------------------------------
    def features(self, symbol: str, live: bool = False) -> pd.DataFrame:
        f = self.base.features(symbol, live)
        nn_f = self._live_forecasts(symbol, f.index) if live else self._research(symbol)
        return f.join(nn_f.reindex(f.index)).astype("float32")

    def _bar_close(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return index + timeframe_delta(self.st.timeframe)

    def _research(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._cache:
            self._build_research()
        return self._cache[symbol]

    def _build_research(self) -> None:
        symbols = [s for s in self.st.all_symbols if self._has_data(s)]
        paths = {s: self._cache_path(s) for s in symbols}
        fresh = all(p.exists() and os.path.getmtime(p) >= os.path.getmtime(self._m1_path(s))
                    for s, p in paths.items())
        if fresh:
            self._cache = {s: pd.read_parquet(p) for s, p in paths.items()}
            return
        log.info("%s: training 1m networks for out-of-sample forecasts (takes a while)...",
                 self.name)
        steps = {s: m1.Steps(m1.five_minute_steps(pd.read_parquet(self._m1_path(s))))
                 for s in symbols}
        index = {s: self.base.features(s).index for s in symbols}
        out, table = m1.research_features(
            steps, {s: self._bar_close(i) for s, i in index.items()}, self.enc_cfg)
        for s in symbols:
            df = pd.DataFrame(out[s], index=index[s], columns=m1.FEATURES)
            paths[s].parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(paths[s])
            self._cache[s] = df
        self.report_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(self.report_dir / "encoders.csv", index=False)

    def _live_forecasts(self, symbol: str, index: pd.DatetimeIndex) -> pd.DataFrame:
        m1_bars = pd.read_parquet(self._m1_path(symbol))
        m1_bars = m1_bars[m1_bars["open_time"] >= m1_bars["open_time"].iloc[-1]
                          - pd.Timedelta(days=LIVE_DAYS)]
        st = m1.Steps(m1.five_minute_steps(m1_bars))
        recent = index[-6:]
        ends = st.end_step(self._bar_close(recent))
        ok = ends >= 0
        df = pd.DataFrame(np.nan, index=recent, columns=m1.FEATURES, dtype="float32")
        if ok.any():
            df.loc[recent[ok]] = self.encoder.forecast(st, ends[ok])
        return df

    # --- research ---------------------------------------------------------------------------
    def walk_forward(self) -> WalkForwardScores:
        wf = super().walk_forward()
        cal_end = pd.Timestamp(self.enc_cfg["calibration_end"], tz="UTC")
        wf.scores = wf.scores[wf.scores.index >= cal_end]
        return wf

    def fit_final(self) -> dict:
        symbols = [s for s in self.st.all_symbols if self._has_data(s)]
        steps = [m1.Steps(m1.five_minute_steps(pd.read_parquet(self._m1_path(s))))
                 for s in symbols]
        now = max(st.index[-1] for st in steps) + m1.STEP
        enc = m1.Encoder(self.enc_cfg).fit([(st, m1._training_ends(st, None, now))
                                            for st in steps])
        version = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        folder = MODELS_DIR / f"{self.name}_encoder" / version
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(enc.state(), folder / "encoder.pt")
        (folder.parent / "current.json").write_text(json.dumps({"version": version}),
                                                    encoding="utf-8")
        return super().fit_final()

    # --- live -------------------------------------------------------------------------------
    def load(self) -> None:
        root = MODELS_DIR / f"{self.name}_encoder"
        version = json.loads((root / "current.json").read_text(encoding="utf-8"))["version"]
        self.encoder = m1.Encoder.from_state(torch.load(root / version / "encoder.pt",
                                                        weights_only=False))
        super().load()
