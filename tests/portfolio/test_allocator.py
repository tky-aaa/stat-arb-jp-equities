import numpy as np
import pandas as pd

from statarb.cointegration.selection import SelectedSpread
from statarb.portfolio.allocator import (
    allocate_inverse_volatility,
)


def test_allocate_inverse_volatility():

    selected_spreads = [
        SelectedSpread(
            tickers=["A", "B"],
            beta=[1.0, -1.0],
            rank=1,
            beta_index=0,
            score=1.0,
            half_life=5.0,
            persistence=0.9,
        ),
        SelectedSpread(
            tickers=["C", "D"],
            beta=[1.0, -1.0],
            rank=1,
            beta_index=0,
            score=2.0,
            half_life=10.0,
            persistence=0.8,
        ),
    ]

    log_prices = pd.DataFrame(
        {
            "A": [10, 11, 10, 12, 11, 10],
            "B": [9.5, 10.2, 9.8, 11.1, 10.4, 9.3],
            "C": [20, 21, 20, 22, 21, 20],
            "D": [19.2, 20.1, 19.4, 21.3, 20.5, 19.1],
        },
        index=pd.date_range(
            "2024-01-01",
            periods=6,
        ),
    )

    result = allocate_inverse_volatility(
        selected_spreads,
        log_prices,
    )

    assert len(result.spreads) == 2

    assert len(result.weights) == 2

    assert np.isclose(
        sum(result.weights),
        1.0,
    )

    assert all(weight > 0 for weight in result.weights)
