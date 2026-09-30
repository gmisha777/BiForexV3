import numpy as np
import pandas as pd

from biforex.config import Settings
from biforex.model import evaluate as ev
from biforex.model.dataset import Dataset, pool
from biforex.model.walkforward import monthly_folds, run_walkforward


def settings() -> Settings:
    s = Settings(symbols=["TEST"])
    s.retrain.train_months = 6
    s.label.horizon_bars = 24
    s.model.params = {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 7,
                      "min_child_samples": 50}
    s.model.early_stopping_rounds = 20
    return s


def synthetic(n_hours: int = 24 * 400, seed: int = 0) -> Dataset:
    """Feature 'signal' shifts the class probabilities; 'noise' is irrelevant."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n_hours, freq="1h", tz="UTC", name="open_time")
    signal = rng.normal(size=n_hours)
    noise = rng.normal(size=n_hours)
    p_up = 1 / (1 + np.exp(-1.5 * signal)) * 0.6
    p_down = (1 - 1 / (1 + np.exp(-1.5 * signal))) * 0.6
    u = rng.uniform(size=n_hours)
    label = np.where(u < p_up, 1, np.where(u < p_up + p_down, -1, 0))
    outcomes = pd.DataFrame({
        "label": label,
        "long_outcome": np.where(label == 1, 1, -1), "long_ret": np.where(label == 1, .02, -.01),
        "long_bars": 5, "short_outcome": np.where(label == -1, 1, -1),
        "short_ret": np.where(label == -1, .02, -.01), "short_bars": 5, "atr": 1.0,
    }, index=idx)
    X = pd.DataFrame({"signal": signal, "noise": noise, "ema21_dist_atr": noise,
                      "ema55_dist_atr": -noise}, index=idx).astype("float32")
    return Dataset(X, pd.Series(label + 1, index=idx, dtype="int8"), outcomes)


def test_folds_purge_and_cover_all_test_months():
    idx = pd.date_range("2022-01-01", "2023-01-15", freq="1h", tz="UTC")
    purge = pd.Timedelta("25h")
    folds = list(monthly_folds(idx, 6, purge))
    assert folds[0].test_start == pd.Timestamp("2022-07-01", tz="UTC")
    assert folds[-1].test_start == pd.Timestamp("2023-01-01", tz="UTC")
    for f in folds:
        # the last training label (open + 24 bars) ends before the test starts
        assert f.train_end + pd.Timedelta("24h") < f.test_start
        assert f.test_end - f.test_start >= pd.Timedelta("28D")
    for a, b in zip(folds, folds[1:], strict=False):
        assert a.test_end == b.test_start


def test_walkforward_pooled_symbols_with_repeated_timestamps():
    ds = pool({"A": synthetic(24 * 300, seed=1), "B": synthetic(24 * 300, seed=2)})
    assert ds.X.index.is_monotonic_increasing
    assert set(ds.X["symbol_id"].unique()) == {0.0, 1.0}
    wf = run_walkforward(ds, settings())
    preds = wf.predictions
    assert len(preds) == wf.folds["test_rows"].sum()
    assert preds.index.duplicated().sum() == len(preds) // 2
    assert preds.groupby("symbol").size().to_dict() == {"A": len(preds) // 2, "B": len(preds) // 2}


def test_walkforward_finds_planted_signal():
    ds = synthetic()
    s = settings()
    wf = run_walkforward(ds, s)
    preds = wf.predictions
    assert preds.index.min() >= pd.Timestamp("2022-07-01", tz="UTC")
    assert not preds.index.duplicated().any()
    assert np.allclose(preds[["p_down", "p_flat", "p_up"]].sum(axis=1), 1, atol=1e-6)

    q = ev.probability_quality(preds, wf.folds)
    assert q.loc["total", "improvement_pct"] > 5
    assert ev.top_features(wf.importance).index[0] == "signal"

    assert len(preds) == wf.folds["test_rows"].sum()

    table = ev.signal_table(preds, cost=0.001)
    assert table.loc[("long", "p>=0.45"), "tp_rate_pct"] > table.loc[("long", "every bar"),
                                                                     "tp_rate_pct"]
