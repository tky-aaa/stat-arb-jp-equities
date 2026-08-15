import pandas as pd

from ..backtest.report import BacktestMetrics
from ..cointegration.selection import SelectedSpread


def create_backtest_summary(
    selected_spreads: list[SelectedSpread],
    metrics: list[BacktestMetrics],
) -> pd.DataFrame:
    """
    Create summary table from selected spreads and backtest metrics.
    """

    if len(selected_spreads) != len(metrics):
        raise ValueError("Number of spreads and metrics must match.")

    records = []

    for spread, metric in zip(
        selected_spreads,
        metrics,
    ):
        records.append(
            {
                "tickers": tuple(spread.tickers),
                "beta": spread.beta,
                "score": spread.score,
                "half_life": spread.half_life,
                "persistence": spread.persistence,
                "total_return": metric.total_return,
                "annualized_return": metric.annualized_return,
                "volatility": metric.volatility,
                "sharpe_ratio": metric.sharpe_ratio,
                "max_drawdown": metric.max_drawdown,
                "win_rate": metric.win_rate,
                "number_of_trades": metric.number_of_trades,
            }
        )

    return pd.DataFrame(records)
