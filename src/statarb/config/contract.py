from dataclasses import dataclass

import numpy as np
import pandas as pd

# Accessible only by orchestrators and APIs
# that receive a given contract.


@dataclass(frozen=True)
class Prices:
    training: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class Group:
    tickers: list[str]


@dataclass(frozen=True)
class SubGroup:
    tickers: list[str]


@dataclass(frozen=True)
class JohansenResult:
    rank: int
    tickers: list[str]
    beta: np.ndarray
    eigenvalues: np.ndarray
    trace_statistics: np.ndarray
    critical_values: np.ndarray
    var_lag: int
    k_ar_diff: int


@dataclass(frozen=True)
class Spread:
    tickers: list[str]
    beta: pd.DataFrame
    intercept: pd.Series
    values: pd.Series


@dataclass(frozen=True)
class SpreadEvaluation:
    # Stationarity
    adf_stat: float
    adf_pvalue: float
    kpss_stat: float
    kpss_pvalue: float

    # Mean reversion
    rho1: float
    phi: float
    half_life: float
    persistence: float

    # Spread distribution
    mean: float
    variance: float
    std: float

    # Higher autocorrelation
    portmanteau: float


@dataclass(frozen=True)
class CointegrationAnalysis:
    rank: int
    beta_index: int
    spread: Spread
    evaluation: SpreadEvaluation


@dataclass(frozen=True)
class Signal:
    spread: pd.Series
    zscore: pd.Series
    position: pd.Series
    beta: pd.DataFrame


@dataclass(frozen=True)
class PortfolioDecision:
    analyses: list[CointegrationAnalysis]
    signals: list[Signal]
    weights: list[float]


@dataclass(frozen=True)
class BacktestResult:
    pnl: pd.Series
    cumulative_pnl: pd.Series
    equity: pd.Series


@dataclass(frozen=True)
class BacktestMetrics:
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    number_of_trades: int


@dataclass(frozen=True)
class LiveStrategy:
    analyses: list[CointegrationAnalysis]
    training_start: str
    training_end: str
