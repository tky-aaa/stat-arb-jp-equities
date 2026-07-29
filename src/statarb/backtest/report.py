from dataclasses import dataclass

import pandas as pd

from .engine import BacktestResult
from .metrics import (
    annualized_return,
    max_drawdown,
    number_of_trades,
    sharpe_ratio,
    total_return,
    volatility,
    win_rate,
)


@dataclass(frozen=True)
class BacktestMetrics:
    """
    Summary metrics of backtest.
    """

    total_return: float

    annualized_return: float

    volatility: float

    sharpe_ratio: float

    max_drawdown: float

    win_rate: float

    number_of_trades: int


def evaluate_backtest(
    result: BacktestResult,
    signal: pd.Series,
    *,
    periods_per_year: int = 252,
) -> BacktestMetrics:
    """
    Calculate backtest performance metrics.
    """

    return BacktestMetrics(
        total_return=total_return(
            result.equity,
        ),
        annualized_return=annualized_return(
            result.equity,
            periods_per_year,
        ),
        volatility=volatility(
            result.returns,
            periods_per_year,
        ),
        sharpe_ratio=sharpe_ratio(
            result.returns,
            periods_per_year,
        ),
        max_drawdown=max_drawdown(
            result.equity,
        ),
        win_rate=win_rate(
            result.returns,
        ),
        number_of_trades=number_of_trades(
            signal,
        ),
    )
