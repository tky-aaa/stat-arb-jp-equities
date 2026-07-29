from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """
    Result of spread strategy backtest.
    """

    returns: pd.Series

    equity: pd.Series


def run_backtest(
    spread: pd.Series,
    signal: pd.Series,
) -> BacktestResult:
    """
    Run simple spread backtest.

    Parameters
    ----------
    spread:
        Spread time series.

    signal:
        Position sizing signal.

        Example:
            +1 : long spread
            -1 : short spread
             0 : flat

    Returns
    -------
    BacktestResult
    """

    if not spread.index.equals(signal.index):
        raise ValueError("spread and signal index must match.")

    # Position decided at t is executed at t+1
    position = signal.shift(1)

    # Spread movement
    spread_change = spread.diff()

    # Strategy return
    returns = position * spread_change

    returns = returns.fillna(0)

    # Equity curve
    equity = (1 + returns).cumprod()

    return BacktestResult(
        returns=returns,
        equity=equity,
    )
