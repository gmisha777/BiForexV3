import numpy as np
import pandas as pd
import pytest

from biforex.config import Settings, StrategyConfig
from biforex.ensemble import combine, combine_frames, entry_side, exit_signal, resolve_weights
from biforex.strategies import REGISTRY
from biforex.strategies.lgbm_tb import probs_to_score


def test_weighted_mean_and_missing_strategy_renormalizes():
    w = {"a": 1.0, "b": 3.0}
    assert combine({"a": 0.4, "b": 0.0}, w, abstain_on_zero=False) == pytest.approx(0.1)
    assert combine({"a": 0.4, "b": None}, w) == pytest.approx(0.4)
    assert combine({"a": None, "b": None}, w) is None


def test_resolve_weights_from_config_and_arguments():
    s = Settings(symbols=["X"], strategies={
        "a": StrategyConfig(weight=1), "b": StrategyConfig(enabled=False),
        "c": StrategyConfig(weight=2)})
    assert resolve_weights(s) == {"a": 1.0, "c": 2.0}
    assert resolve_weights(s, ["b"]) == {"b": 1.0}
    assert resolve_weights(s, ["a", "b"], [1, 3]) == {"a": 1.0, "b": 3.0}
    with pytest.raises(ValueError):
        resolve_weights(s, ["a", "b"], [1])


def test_zero_score_abstains_by_default():
    w = {"a": 1.0, "b": 1.0}
    assert combine({"a": 0.2, "b": 0.0}, w) == pytest.approx(0.2)
    assert combine({"a": 0.2, "b": 0.0}, w, abstain_on_zero=False) == pytest.approx(0.1)
    assert combine({"a": 0.0, "b": 0.0}, w) == 0.0  # everyone abstained: no view, not missing
    assert combine({"a": 0.0, "b": None}, w) == 0.0
    assert combine({"a": None, "b": None}, w) is None


def test_combine_frames_aligns_by_time_and_symbol():
    t = pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC", name="open_time")
    a = pd.DataFrame({"symbol": ["X", "X"], "score": [0.2, 0.4]}, index=t)
    b = pd.DataFrame({"symbol": ["X"], "score": [-0.2]}, index=t[:1])
    out = combine_frames({"a": a, "b": b}, {"a": 1.0, "b": 1.0}, abstain_on_zero=False)
    assert out["score"].tolist() == pytest.approx([0.0, 0.4])
    assert out["n_strategies"].tolist() == [2, 1]


def test_entry_and_exit_rules():
    assert entry_side(0.12, 0.10) == 1 and entry_side(-0.12, 0.10) == -1
    assert entry_side(0.05, 0.10) == 0
    assert exit_signal(1, -0.15, 0.10) and not exit_signal(1, 0.5, 0.10)
    assert exit_signal(-1, 0.10, 0.10) and not exit_signal(-1, 0.05, 0.10)


def test_lgbm_score_matches_old_probability_rule():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([2, 3, 2], size=5000)
    p_down, p_up = p[:, 0], p[:, 2]
    score = probs_to_score(p_down, p_up)
    old_long = (p_up >= 0.40) & (p_up > p_down)
    old_short = (p_down >= 0.40) & (p_down > p_up)
    assert ((score >= 0.10 - 1e-12) == old_long).all()
    assert ((score <= -0.10 + 1e-12) == old_short).all()
    assert score.min() >= -1 and score.max() <= 1


def test_lgbm_strategy_registered():
    assert "lgbm_tb" in REGISTRY
