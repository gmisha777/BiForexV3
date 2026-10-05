import numpy as np
import pandas as pd
import pytest

from biforex.features.flow import flow_features


def synth_m1(days: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 1440
    t = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 5e-4, n)))
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.uniform(1, 10, n)
    return pd.DataFrame({
        "open_time": t, "open": open_, "high": np.maximum(open_, close) * 1.0002,
        "low": np.minimum(open_, close) * 0.9998, "close": close, "volume": vol,
        "trades": rng.integers(5, 50, n), "taker_buy_volume": vol * rng.uniform(0.2, 0.8, n),
    })


@pytest.mark.parametrize("cut_minutes", [12 * 1440 + 100, 12 * 1440 + 240, 15 * 1440 + 17])
def test_flow_features_no_lookahead(cut_minutes):
    m1 = synth_m1(20)
    full = flow_features(m1)
    past = flow_features(m1.iloc[:cut_minutes])
    last_minute = m1["open_time"].iloc[cut_minutes - 1]
    bar_start = last_minute.floor("4h")
    if last_minute + pd.Timedelta("1min") != bar_start + pd.Timedelta("4h"):
        past = past[past.index < bar_start]  # the bar still being formed is not closed yet
    pd.testing.assert_frame_equal(past, full.loc[past.index], check_freq=False)


def test_flow_bar_values():
    m1 = synth_m1(3)
    f = flow_features(m1)
    assert len(f) == 18  # 3 days x 6 bars
    assert f["close_pos_range"].between(0, 1).all()
    assert f["t_high"].between(0, 1).all()
    assert f["taker_ratio"].between(0.2, 0.8).all()
