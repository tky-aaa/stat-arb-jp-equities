import numpy as np
import pandas as pd


def total_return(
    equity: pd.Series,
) -> float:
    """
    Total return.
    """

    initial = equity.iloc[0]

    if initial == 0:
        return float(equity.iloc[-1] - initial)

    return float(equity.iloc[-1] / initial - 1)


def annualized_return(
    equity: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized return.
    """

    total = total_return(equity)

    periods = len(equity)

    return float((1 + total) ** (periods_per_year / periods) - 1)


def volatility(
    pnl: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized volatility.
    """

    return float(pnl.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    pnl: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized Sharpe ratio.
    """

    vol = volatility(
        pnl,
        periods_per_year,
    )

    if vol == 0:
        return 0.0

    return float(pnl.mean() * periods_per_year / vol)


def max_drawdown(
    equity: pd.Series,
) -> float:
    """
    Maximum drawdown.
    """

    running_max = equity.cummax()

    drawdown = (equity - running_max) / running_max

    return float(drawdown.min())


def win_rate(
    pnl: pd.Series,
) -> float:
    """
    Fraction of profitable periods.
    """

    trades = pnl[pnl != 0]

    if len(trades) == 0:
        return 0.0

    return float((trades > 0).mean())


def number_of_trades(
    signal: pd.Series,
) -> int:
    """
    Count position changes.
    """

    return int((signal.diff().abs() > 0).sum())
