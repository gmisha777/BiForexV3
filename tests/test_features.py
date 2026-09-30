import numpy as np
import pandas as pd
import pytest

from biforex.features import indicators as ind
from biforex.features.pipeline import compute_features

START = pd.Timestamp("2021-01-01", tz="UTC")


def synth_klines(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range(START, periods=n, freq="1h")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n))
    volume = rng.uniform(100, 200, n)
    taker = volume * rng.uniform(0.3, 0.7, n)
    return pd.DataFrame({
        "open_time": t, "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "close_time": t + pd.Timedelta("1h") - pd.Timedelta("1ms"),
        "quote_volume": volume * close, "trades": rng.integers(50, 500, n),
        "taker_buy_volume": taker, "taker_buy_quote_volume": taker * close,
    })


def synth_funding(n_hours: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range(START, periods=n_hours // 8, freq="8h")
    return pd.DataFrame({"funding_time": t, "funding_rate": rng.normal(1e-4, 1e-4, len(t))})


def synth_metrics(n_hours: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range(START + pd.Timedelta("30D"), periods=(n_hours - 720) * 12, freq="5min")
    n = len(t)
    return pd.DataFrame({
        "time": t,
        "sum_open_interest": 1e4 * np.exp(np.cumsum(rng.normal(0, 1e-3, n))),
        "sum_open_interest_value": 1e6 * np.exp(np.cumsum(rng.normal(0, 1e-3, n))),
        "count_toptrader_long_short_ratio": rng.uniform(0.8, 1.5, n),
        "sum_toptrader_long_short_ratio": rng.uniform(0.8, 1.5, n),
        "count_long_short_ratio": rng.uniform(0.8, 1.5, n),
        "sum_taker_long_short_vol_ratio": rng.uniform(0.5, 2.0, n),
    })


N = 2400


@pytest.fixture(scope="module")
def sources():
    return {
        "klines": synth_klines(N, 1),
        "context": synth_klines(N, 2),
        "funding": synth_funding(N, 3),
        "metrics": synth_metrics(N, 4),
    }


def truncate(sources: dict, bars: int) -> dict:
    """What the bot would know right after bar `bars - 1` closes."""
    decision = sources["klines"]["open_time"].iloc[bars - 1] + pd.Timedelta("1h")
    return {
        "klines": sources["klines"].iloc[:bars],
        "context": sources["context"].iloc[:bars],
        "funding": sources["funding"][sources["funding"]["funding_time"] <= decision],
        "metrics": sources["metrics"][sources["metrics"]["time"] <= decision],
    }


# cut points mid-4h-bar, mid-day, right at a funding time, right after a day closes
@pytest.mark.parametrize("bars", [1234, 1500, 1601, 1680])
def test_no_lookahead(sources, bars):
    full = compute_features(**sources)
    past = compute_features(**truncate(sources, bars))
    assert list(past.columns) == list(full.columns)
    pd.testing.assert_frame_equal(past, full.iloc[:bars], check_freq=False)


def test_all_feature_groups_present(sources):
    f = compute_features(**sources)
    for prefix in ("ret_", "ema", "rsi", "h4h_", "h1d_", "funding_", "oi_", "ctx_", "hour_"):
        assert any(c.startswith(prefix) for c in f.columns), prefix
    tail = f.iloc[-500:]
    assert tail.notna().all().all(), tail.columns[tail.isna().any()].tolist()


def test_optional_sources_can_be_missing(sources):
    f = compute_features(sources["klines"])
    assert not any(c.startswith(("funding_", "oi_", "ctx_")) for c in f.columns)


def test_rsi_bounds_and_atr_constant_range():
    k = synth_klines(500, 5)
    r = ind.rsi(k["close"]).dropna()
    assert r.between(0, 100).all()
    flat = pd.Series(np.full(100, 10.0))
    atr = ind.atr(flat + 1, flat - 1, flat, 14).dropna()
    assert np.allclose(atr, 2.0)
