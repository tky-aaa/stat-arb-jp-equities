import pandas as pd

from statarb.backtest.report import BacktestMetrics
from statarb.cointegration.selection import SelectedSpread
from statarb.reports.summary import (
    create_backtest_summary,
)


def test_create_backtest_summary():

    selected_spreads = [
        SelectedSpread(
            tickers=[
                "A",
                "B",
            ],
            beta=[
                1.0,
                -1.0,
            ],
            rank=1,
            beta_index=0,
            score=1.0,
            half_life=5.0,
            persistence=0.9,
        )
    ]

    metrics = [
        BacktestMetrics(
            total_return=0.1,
            annualized_return=0.2,
            volatility=0.05,
            sharpe_ratio=2.0,
            max_drawdown=-0.03,
            win_rate=0.6,
            number_of_trades=10,
        )
    ]

    result = create_backtest_summary(
        selected_spreads,
        metrics,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert len(result) == 1

    assert tuple(result.iloc[0]["tickers"]) == (
        "A",
        "B",
    )

    assert result.iloc[0]["sharpe_ratio"] == 2.0
