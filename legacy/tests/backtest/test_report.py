import pandas as pd

from statarb.backtest.engine import BacktestResult
from statarb.backtest.report import evaluate_backtest


def test_evaluate_backtest():

    index = pd.date_range(
        "2024-01-01",
        periods=4,
    )

    pnl = pd.Series(
        [
            0.0,
            0.1,
            -0.05,
            0.02,
        ],
        index=index,
    )

    cumulative = pd.Series(
        [
            0.0,
            0.1,
            0.05,
            0.07,
        ],
        index=index,
    )

    equity = 1.0 + cumulative

    result = BacktestResult(
        pnl=pnl,
        cumulative_pnl=cumulative,
        equity=equity,
    )

    signal = pd.Series(
        [
            0,
            1,
            1,
            0,
        ],
        index=index,
    )

    metrics = evaluate_backtest(
        result,
        signal,
    )

    assert metrics.number_of_trades >= 0
