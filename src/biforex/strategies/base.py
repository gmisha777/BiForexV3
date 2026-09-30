"""Strategy interface.

A strategy is one decision approach (a LightGBM model, a neural network, a rule, ...). All
strategies share the data, the decision timeframe (settings.timeframe) and the exit rules
(stop / take profit / time limit from settings.label). Each gives, for every symbol at every
bar close, a score in [-1, +1]: -1 strongly down, 0 no view, +1 strongly up. The ensemble
combines the scores of the enabled strategies with their weights.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd

from ..config import PROJECT_ROOT, Settings


@dataclass
class WalkForwardScores:
    """Out-of-sample scores from walk-forward validation (never seen in training)."""

    scores: pd.DataFrame  # index open_time; columns: symbol, score
    reports: dict[str, pd.DataFrame] = field(default_factory=dict)  # strategy-specific
    summary: str = ""  # human-readable notes for the training log


class Strategy(ABC):
    name: ClassVar[str]

    def __init__(self, settings: Settings, params: dict | None = None):
        self.settings = settings
        self.params = params or {}

    @property
    def report_dir(self):
        return PROJECT_ROOT / "reports" / self.name

    # --- research ------------------------------------------------------------------------
    @abstractmethod
    def walk_forward(self) -> WalkForwardScores:
        """Train on rolling windows and score the following, unseen period."""

    @abstractmethod
    def fit_final(self) -> dict:
        """Train on the most recent data and save the model for live use; returns its meta."""

    # --- live ------------------------------------------------------------------------------
    @abstractmethod
    def load(self) -> None:
        """Load the saved model (called once when the bot starts)."""

    @abstractmethod
    def score(self, symbol: str, bar: pd.Timestamp) -> float | None:
        """Score for the bar with open time `bar` (already closed); None if unavailable."""


def clip_score(x: float) -> float:
    return max(-1.0, min(1.0, float(x)))
