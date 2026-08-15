import pandas as pd
import pytest

from statarb.backtest.metrics import (
    max_drawdown,
    number_of_trades,
    total_return,
    win_rate,
)


def test_total_return():

    equity = pd.Series(
        [
            1.0,
            1.1,
            1.2,
        ]
    )

    assert total_return(equity) == pytest.approx(0.2)


def test_max_drawdown():

    equity = pd.Series(
        [
            1.0,
            1.2,
            0.9,
            1.1,
        ]
    )

    assert (
        round(
            max_drawdown(equity),
            3,
        )
        == -0.25
    )


def test_win_rate():

    pnl = pd.Series(
        [
            0,
            1,
            -1,
            2,
            0,
        ]
    )

    assert win_rate(pnl) == 2 / 3


def test_number_of_trades():

    signal = pd.Series(
        [
            0,
            1,
            1,
            0,
            -1,
        ]
    )

    assert number_of_trades(signal) == 3
