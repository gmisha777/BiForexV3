import numpy as np
import pandas as pd

from biforex.strategies import REGISTRY
from biforex.strategies.som_regime import DEFAULTS, RegimeModel
from biforex.strategies.trend_regime import raw_signal


def test_all_strategies_registered():
    assert {"lgbm_tb", "flow_1m", "trend_regime", "crowd", "som_regime"} <= set(REGISTRY)
    assert REGISTRY["flow_1m"].data_intervals == ("1m",)


def test_trend_raw_signal_sign_and_regime_filter():
    t = pd.date_range("2024-01-01", periods=600, freq="4h", tz="UTC")
    up = np.linspace(100, 200, 600)
    k = pd.DataFrame({"high": up * 1.002, "low": up * 0.998, "close": up}, index=t)
    raw = raw_signal(k, 21, 89, 20).dropna()
    assert (raw.iloc[-100:] > 0).all()  # steady uptrend -> positive
    flat = pd.DataFrame({"high": 101.0, "low": 99.0, "close": 100.0 + np.sin(np.arange(600))},
                        index=t)
    assert (raw_signal(flat, 21, 89, 20).dropna().abs() < 1e-9).mean() > 0.9  # no trend -> 0


def test_regime_model_shapes_and_probabilities():
    rng = np.random.default_rng(0)
    n = 3000
    cols = DEFAULTS["regime_features"] + ["x1", "x2"]
    X = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols).astype("float32")
    y = pd.Series(rng.integers(0, 3, n))
    cfg = {**DEFAULTS, "min_rows": 200, "som_iterations": 500,
           "lgbm": {**DEFAULTS["lgbm"], "n_estimators": 10}}
    m = RegimeModel(cfg).fit(X, y)
    p = m.predict_proba(X.iloc[:50])
    assert p.shape == (50, 3)
    assert np.allclose(p.sum(1), 1)
    assert m.clusters(X).max() < cfg["grid"] ** 2


def _features(n=300, shock_at=250, move=0.05, vol_spike=True):
    t = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    f = pd.DataFrame({"atr_pct": 0.01, "rsi14": 50.0, "oi_chg_4": 0.0,
                      "vol_rel24": 0.0, "ret_1": 0.0, "ret_2": 0.0, "ret_3": 0.0}, index=t)
    f.iloc[shock_at, f.columns.get_loc("ret_1")] = move
    if vol_spike:
        f.iloc[shock_at, f.columns.get_loc("vol_rel24")] = 1.0
    return f


def test_shock_signal_events_and_direction():
    from biforex.strategies.shock_move import shock_signal
    f = _features(move=0.05)  # +5 ATR pump with a volume spike
    raw = shock_signal(f, 2.0, "volume", rsi_filter=False)
    assert raw.iloc[250] == -2.5  # fade direction: short after a pump
    assert (raw.drop(raw.index[250]) == 0).all()
    assert (shock_signal(_features(vol_spike=False), 2.0, "volume", False) == 0).all()


def test_squeeze_signal_breaks_out_after_compression():
    from biforex.strategies.squeeze_breakout import squeeze_signal
    t = pd.date_range("2024-01-01", periods=400, freq="4h", tz="UTC")
    rng = np.random.default_rng(1)
    c = pd.Series(100 + np.cumsum(rng.normal(0, 1, 400)), index=t)
    c.iloc[300:360] = c.iloc[299]  # flat -> compressed volatility
    c.iloc[360] = c.iloc[299] + 5  # breakout up
    k = pd.DataFrame({"high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1.0}, index=t)
    k.iloc[300:360, k.columns.get_loc("high")] = c.iloc[299] + 0.01
    k.iloc[300:360, k.columns.get_loc("low")] = c.iloc[299] - 0.01
    raw = squeeze_signal(k, 0.2, 20, volume_confirm=False)
    assert raw.iloc[360] > 1
