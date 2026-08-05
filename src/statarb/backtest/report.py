from dataclasses import dataclass

import pandas as pd

from .engine import BacktestResult
from .metrics import (
    annualized_return,
    max_drawdown,
    number_of_trades,
    sharpe_ratio,
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


def _build_equity_curve(
    result: BacktestResult,
) -> pd.Series:
    """
    Convert cumulative pnl into equity curve.

    Initial capital is normalized to 1.
    """

    return 1.0 + result.cumulative_pnl


def evaluate_backtest(
    result: BacktestResult,
    signal: pd.Series,
    *,
    periods_per_year: int = 252,
) -> BacktestMetrics:
    """
    Calculate backtest performance metrics.
    """

    equity = _build_equity_curve(
        result,
    )

    return BacktestMetrics(
        total_return=float(equity.iloc[-1] - 1.0),
        annualized_return=annualized_return(
            equity,
            periods_per_year,
        ),
        volatility=volatility(
            result.pnl,
            periods_per_year,
        ),
        sharpe_ratio=sharpe_ratio(
            result.pnl,
            periods_per_year,
        ),
        max_drawdown=max_drawdown(
            equity,
        ),
        win_rate=win_rate(
            result.pnl,
        ),
        number_of_trades=number_of_trades(
            signal,
        ),
    )
