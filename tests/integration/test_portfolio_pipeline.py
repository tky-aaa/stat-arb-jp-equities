import pandas as pd

from statarb.cointegration.selection import (
    SelectedSpread,
)
from statarb.portfolio.allocator import (
    allocate_inverse_volatility,
)
from statarb.portfolio.engine import (
    run_portfolio_backtest,
)


def test_full_portfolio_pipeline():

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
            "A": [10, 11, 10.5, 12, 11, 10],
            "B": [9, 10, 9.8, 11, 10.2, 9.5],
            "C": [20, 21.5, 20, 22, 21, 19.5],
            "D": [19, 20.2, 19.5, 21, 20.3, 19],
        },
        index=pd.date_range(
            "2024-01-01",
            periods=6,
        ),
    )

    portfolio = allocate_inverse_volatility(
        selected_spreads,
        log_prices,
    )

    assert len(portfolio.weights) == 2

    assert abs(sum(portfolio.weights) - 1.0) < 1e-12

    result = run_portfolio_backtest(
        portfolio.spreads,
        log_prices,
        portfolio.weights,
        window=3,
        entry_threshold=1.0,
        exit_threshold=0.5,
    )

    assert len(result.returns) == len(log_prices)

    assert result.equity.notna().all()
