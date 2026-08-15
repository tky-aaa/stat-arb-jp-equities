import pandas as pd

from statarb.backtest.plots import (
    plot_drawdown,
    plot_equity_curve,
    plot_returns,
)


def test_plot_equity_curve():

    equity = pd.Series(
        [
            1.0,
            1.1,
            1.05,
        ]
    )

    fig, ax = plot_equity_curve(
        equity,
    )

    assert fig is not None
    assert ax is not None


def test_plot_drawdown():

    equity = pd.Series(
        [
            1.0,
            1.2,
            0.9,
        ]
    )

    fig, ax = plot_drawdown(
        equity,
    )

    assert fig is not None
    assert ax is not None


def test_plot_returns():

    returns = pd.Series(
        [
            0.1,
            -0.05,
        ]
    )

    fig, ax = plot_returns(
        returns,
    )

    assert fig is not None
    assert ax is not None
