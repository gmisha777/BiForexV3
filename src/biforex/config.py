"""Configuration loading and validation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


class DataConfig(BaseModel):
    dir: Path = Path("data")
    history_start: date = date(2020, 1, 1)
    intervals: list[str] = Field(default_factory=lambda: ["1h"])
    funding: bool = True
    metrics: bool = True
    download_workers: int = 8


class FeaturesConfig(BaseModel):
    context_symbol: str = "BTCUSDT"  # market leader added as context to the other symbols
    higher_timeframes: list[str] = Field(default_factory=lambda: ["1d"])
    metrics_max_staleness: str = "3h"  # older metrics are treated as missing
    warmup_bars: int = 700  # dropped at the start (EMA 610 needs history)


class LabelConfig(BaseModel):
    horizon_bars: int = 12
    tp_atr: float = 2.0
    sl_atr: float = 1.0
    atr_period: int = 14
    intrabar_interval: str = "1m"  # resolves bars where both barriers were touched


class ModelConfig(BaseModel):
    pooled: bool = True  # one model for all symbols, symbol_id as a feature
    extra_symbols: list[str] = Field(default_factory=list)  # used for training only
    val_fraction: float = 0.15  # tail of each training window used for early stopping
    early_stopping_rounds: int = 300
    params: dict = Field(default_factory=lambda: {
        "n_estimators": 3000,
        "learning_rate": 0.01,
        "num_leaves": 8,
        "min_child_samples": 300,
        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.5,
        "reg_lambda": 10.0,
    })


class SignalConfig(BaseModel):
    enter_prob: float = 0.40
    exit_prob: float = 0.40


class BacktestConfig(BaseModel):
    initial_equity: float = 5000.0
    threshold_grid: list[float] = Field(default_factory=lambda: [0.36, 0.38, 0.40, 0.42, 0.45])


class RiskConfig(BaseModel):
    per_trade_pct: float = 1.0
    max_trade_risk_pct: float = 5.0
    max_leverage: float = 5
    daily_loss_pct: float = 6.0
    max_dd_pct: float = 30.0


class BotConfig(BaseModel):
    poll_seconds: int = 30
    bar_close_delay_seconds: int = 20
    metrics_rest_hours: int = 40


class ExecutionConfig(BaseModel):
    testnet: bool = True
    fee_taker: float = 0.0005
    fee_maker: float = 0.0002
    slippage_bps: float = 2
    entry_order: str = "limit"  # "limit" (maker, may not fill) | "market"
    limit_offset_atr: float = 0.1  # limit price this many ATR better than the signal close
    take_profit_limit: bool = True  # TP as a resting limit order (maker)
    exchange_leverage: int = 10  # isolated-margin setting: margin = notional / this


class RetrainConfig(BaseModel):
    every_days: int = 7
    train_months: int = 36


class LoggingConfig(BaseModel):
    level: str = "INFO"
    dir: Path = Path("logs")


class Settings(BaseModel):
    symbols: list[str]
    timeframe: str = "4h"
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    label: LabelConfig = Field(default_factory=LabelConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    retrain: RetrainConfig = Field(default_factory=RetrainConfig)
    bot: BotConfig = Field(default_factory=BotConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @property
    def all_symbols(self) -> list[str]:
        """Traded symbols first, then symbols used only to train the pooled model."""
        return self.symbols + [s for s in self.model.extra_symbols if s not in self.symbols]

    def resolve_paths(self, root: Path) -> Settings:
        """Make relative directories absolute against the project root."""
        if not self.data.dir.is_absolute():
            self.data.dir = root / self.data.dir
        if not self.logging.dir.is_absolute():
            self.logging.dir = root / self.logging.dir
        return self


def load_settings(path: Path | str | None = None) -> Settings:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    load_dotenv(path.parent / ".env")
    return Settings.model_validate(raw).resolve_paths(PROJECT_ROOT)
