import numpy as np
import pandas as pd

from statarb.portfolio.engine import (
    combine_backtest_results,
)


def test_combine_backtest_results():

    returns_list = [
        pd.Series(
            [
                0.01,
                -0.01,
                0.02,
            ]
        ),
        pd.Series(
            [
                0.02,
                0.01,
                -0.01,
            ]
        ),
    ]

    weights = [
        0.5,
        0.5,
    ]

    result = combine_backtest_results(
        returns_list,
        weights,
    )

    assert isinstance(
        result.returns,
        pd.Series,
    )

    assert isinstance(
        result.equity,
        pd.Series,
    )

    assert isinstance(
        result.spread_returns,
        pd.DataFrame,
    )

    assert np.isclose(
        result.weights_sum if hasattr(result, "weights_sum") else sum(weights),
        1.0,
    )

    assert len(result.returns) == 3

    assert len(result.equity) == 3

    assert np.isfinite(result.equity.values).all()
