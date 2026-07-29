# src/statarb/backtest/plots.py

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(
    equity: pd.Series,
):
    """
    Plot equity curve.
    """

    fig, ax = plt.subplots(
        figsize=(10, 4),
    )

    ax.plot(
        equity.index,
        equity.values,
    )

    ax.set_title(
        "Equity Curve",
    )

    ax.set_xlabel(
        "Date",
    )

    ax.set_ylabel(
        "Equity",
    )

    ax.grid(
        True,
    )

    return fig, ax


def plot_drawdown(
    equity: pd.Series,
):
    """
    Plot drawdown curve.
    """

    running_max = equity.cummax()

    drawdown = (equity - running_max) / running_max

    fig, ax = plt.subplots(
        figsize=(10, 4),
    )

    ax.plot(
        drawdown.index,
        drawdown.values,
    )

    ax.set_title(
        "Drawdown",
    )

    ax.set_xlabel(
        "Date",
    )

    ax.set_ylabel(
        "Drawdown",
    )

    ax.grid(
        True,
    )

    return fig, ax


def plot_returns(
    returns: pd.Series,
):
    """
    Plot period returns.
    """

    fig, ax = plt.subplots(
        figsize=(10, 4),
    )

    ax.bar(
        returns.index,
        returns.values,
    )

    ax.set_title(
        "Returns",
    )

    ax.set_xlabel(
        "Date",
    )

    ax.set_ylabel(
        "Return",
    )

    ax.grid(
        True,
    )

    return fig, ax
