import numpy as np
import pandas as pd

from statarb.backtest.pipeline import (
    run_spread_backtest,
)
from statarb.cointegration.selection import (
    SelectedSpread,
)


def test_kalman_spread_pipeline():

    selected_spread = SelectedSpread(
        tickers=[
            "A",
            "B",
            "C",
        ],
        beta=[
            1.0,
            -1.0,
            0.5,
        ],
        rank=1,
        beta_index=0,
        score=1.0,
        half_life=5.0,
        persistence=0.9,
    )

    log_prices = pd.DataFrame(
        {
            "A": [
                20.0,
                21.0,
                22.0,
                21.5,
                22.5,
                23.0,
            ],
            "B": [
                10.0,
                10.5,
                11.0,
                10.8,
                11.2,
                11.5,
            ],
            "C": [
                5.0,
                5.2,
                5.5,
                5.4,
                5.6,
                5.8,
            ],
        },
        index=pd.date_range(
            "2024-01-01",
            periods=6,
        ),
    )

    result = run_spread_backtest(
        selected_spread,
        log_prices,
        window=3,
        entry_threshold=1.0,
        exit_threshold=0.5,
        spread_method="kalman",
    )

    assert isinstance(
        result.spread,
        pd.Series,
    )

    assert isinstance(
        result.zscore,
        pd.Series,
    )

    assert isinstance(
        result.signal,
        pd.Series,
    )

    assert len(result.spread) == len(log_prices)

    assert len(result.zscore) == len(log_prices)

    assert len(result.signal) == len(log_prices)

    assert len(result.backtest.pnl) == len(log_prices)

    assert len(result.backtest.cumulative_pnl) == len(log_prices)

    assert np.isfinite(
        result.spread.values,
    ).all()

    assert result.backtest.cumulative_pnl.notna().all()
