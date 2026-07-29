import pandas as pd

from statarb.cointegration.selection import (
    SelectedSpread,
)
from statarb.portfolio.engine import (
    run_portfolio_backtest,
)


def test_run_portfolio_backtest():

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
        ),
        SelectedSpread(
            tickers=[
                "C",
                "D",
            ],
            beta=[
                1.0,
                -1.0,
            ],
            rank=1,
            beta_index=0,
            score=2.0,
            half_life=10.0,
            persistence=0.8,
        ),
    ]

    log_prices = pd.DataFrame(
        {
            "A": [
                10,
                11,
                10,
                12,
                11,
                10,
            ],
            "B": [
                9,
                10,
                9,
                11,
                10,
                9,
            ],
            "C": [
                20,
                21,
                20,
                22,
                21,
                20,
            ],
            "D": [
                19,
                20,
                19,
                21,
                20,
                19,
            ],
        },
        index=pd.date_range(
            "2024-01-01",
            periods=6,
        ),
    )

    result = run_portfolio_backtest(
        selected_spreads,
        log_prices,
        weights=[
            0.5,
            0.5,
        ],
        window=3,
        entry_threshold=1.0,
        exit_threshold=0.5,
    )

    assert len(result.returns) == len(log_prices)

    assert len(result.equity) == len(log_prices)

    assert result.spread_returns.shape == (
        len(log_prices),
        2,
    )

    assert result.equity.notna().all()
