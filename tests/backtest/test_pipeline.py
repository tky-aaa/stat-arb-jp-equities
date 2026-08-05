import pandas as pd

from statarb.backtest.pipeline import run_spread_backtest
from statarb.cointegration.selection import SelectedSpread


def test_run_strategy_backtest():

    selected_spread = SelectedSpread(
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

    log_prices = pd.DataFrame(
        {
            "A": [
                10.0,
                11.0,
                10.5,
                12.0,
                11.0,
                10.0,
            ],
            "B": [
                9.0,
                10.0,
                9.5,
                11.0,
                10.5,
                9.5,
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
