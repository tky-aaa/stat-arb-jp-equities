import pandas as pd

from statarb.backtest.engine import (
    BacktestResult,
)
from statarb.backtest.report import (
    evaluate_backtest,
)


def test_evaluate_backtest():

    index = pd.date_range(
        "2024-01-01",
        periods=4,
    )

    result = BacktestResult(
        returns=pd.Series(
            [
                0.0,
                0.1,
                -0.05,
                0.02,
            ],
            index=index,
        ),
        equity=pd.Series(
            [
                1.0,
                1.1,
                1.045,
                1.0659,
            ],
            index=index,
        ),
    )

    signal = pd.Series(
        [
            0,
            1,
            -1,
            0,
        ],
        index=index,
    )

    metrics = evaluate_backtest(
        result,
        signal,
    )

    assert metrics.total_return > 0

    assert metrics.max_drawdown < 0

    assert metrics.number_of_trades == 3
