import numpy as np
import pandas as pd
import pytest

from biforex.labels.triple_barrier import SL, TIMEOUT, TP, triple_barrier

T0 = pd.Timestamp("2024-01-01", tz="UTC")
H = pd.Timedelta("1h")


def bars(rows: list[tuple[float, float, float, float]], freq: str = "1h") -> pd.DataFrame:
    t = pd.date_range(T0, periods=len(rows), freq=freq)
    o, h, lo, c = map(np.array, zip(*rows, strict=True))
    return pd.DataFrame({"open_time": t, "open": o, "high": h, "low": lo, "close": c})


def flat(n: int, price: float = 100.0) -> list[tuple]:
    """Bars with range 2 -> ATR = 2 once warmed up."""
    return [(price, price + 1, price - 1, price)] * n


WARM = 20  # ATR(14) warm-up; the bar under test is WARM - 1 (entry 100, ATR 2)


def label_row(future: list[tuple], intrabar=None, horizon=5, **kw) -> pd.Series:
    df = bars(flat(WARM) + future + flat(horizon))
    out = triple_barrier(df, horizon=horizon, tp_atr=2, sl_atr=1, atr_period=14,
                         intrabar=intrabar, **kw)
    return out.iloc[WARM - 1]


def test_long_take_profit():
    r = label_row([(100, 101, 99.5, 101), (101, 104.5, 100.5, 104)])  # TP 104, SL 98
    assert r["atr"] == pytest.approx(2.0)
    assert (r["label"], r["long_outcome"], r["short_outcome"]) == (1, TP, SL)
    assert r["long_ret"] == pytest.approx(0.04) and r["long_bars"] == 2
    assert r["short_ret"] == pytest.approx(-0.02) and r["short_bars"] == 2


def test_short_take_profit():
    r = label_row([(100, 100.5, 95.5, 96)])  # short TP 96, short SL 102
    assert (r["label"], r["short_outcome"], r["long_outcome"]) == (-1, TP, SL)
    assert r["short_ret"] == pytest.approx(0.04)


def test_timeout_uses_close_at_horizon():
    r = label_row(flat(3, 100) + [(100, 101, 99, 101.5)] * 2, horizon=5)
    assert (r["label"], r["long_outcome"], r["short_outcome"]) == (0, TIMEOUT, TIMEOUT)
    assert r["long_ret"] == pytest.approx(0.015) and r["short_ret"] == pytest.approx(-0.015)
    assert r["long_bars"] == 5


def test_both_in_one_bar_without_intrabar_is_stop():
    r = label_row([(100, 105, 97, 100)])  # touches long TP 104 and long SL 98
    assert r["long_outcome"] == SL


@pytest.mark.parametrize("tp_minute_first, expected", [(True, TP), (False, SL)])
def test_both_in_one_bar_resolved_by_minutes(tp_minute_first, expected):
    hour = T0 + WARM * H  # the first future bar
    up, down = (100, 105, 99.5, 104), (100, 100.5, 97, 98)
    minutes = [up, down] if tp_minute_first else [down, up]
    fine = bars(minutes + [(100, 100.1, 99.9, 100)] * 58, freq="1min")
    fine["open_time"] = fine["open_time"] - T0 + hour
    r = label_row([(100, 105, 97, 100)], intrabar=fine)
    assert r["long_outcome"] == expected


def test_same_minute_is_stop():
    hour = T0 + WARM * H
    fine = bars([(100, 105, 97, 100)] * 60, freq="1min")
    fine["open_time"] = fine["open_time"] - T0 + hour
    assert label_row([(100, 105, 97, 100)], intrabar=fine)["long_outcome"] == SL


def test_tail_and_warmup_are_nan():
    df = bars(flat(40))
    out = triple_barrier(df, horizon=5, atr_period=14)
    assert out["label"].iloc[-5:].isna().all()
    assert out["label"].iloc[:13].isna().all()
    assert out["label"].iloc[13:-5].notna().all()


def test_random_walk_outcomes_consistent():
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 3000)))
    o = np.r_[c[0], c[:-1]]
    df = bars(list(zip(o, np.maximum(o, c) * 1.003, np.minimum(o, c) * 0.997, c, strict=True)))
    out = triple_barrier(df, horizon=24).dropna(subset=["label"])
    both_tp = (out["long_outcome"] == TP) & (out["short_outcome"] == TP)
    assert not both_tp.any()
    assert set(out["label"].unique()) <= {-1, 0, 1}
    assert (out.loc[out["long_outcome"] == TP, "long_ret"] > 0).all()
    assert (out.loc[out["long_outcome"] == SL, "long_ret"] < 0).all()


def test_rejects_non_exclusive_barriers():
    with pytest.raises(ValueError):
        triple_barrier(bars(flat(40)), tp_atr=1, sl_atr=1)


def test_window_across_a_gap_is_not_labelled():
    df = bars(flat(40))
    df = df.drop(index=range(30, 33)).reset_index(drop=True)  # 3 missing hours
    out = triple_barrier(df, horizon=5, atr_period=14)
    t = out.index
    gap_start = pd.Timestamp(T0 + 29 * H)
    crosses = (t < gap_start + H) & (t >= gap_start - 4 * H)
    assert out.loc[crosses, "label"].isna().all()
    assert out.loc[(t >= T0 + 13 * H) & (t < gap_start - 4 * H), "label"].notna().all()
