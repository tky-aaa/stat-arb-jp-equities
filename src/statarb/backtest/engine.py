from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """
    Result of spread strategy backtest.
    """

    pnl: pd.Series

    cumulative_pnl: pd.Series

    equity: pd.Series


def run_backtest(
    prices: pd.DataFrame,
    weights: np.ndarray,
    signal: pd.Series,
    *,
    initial_capital: float = 1.0,
) -> BacktestResult:
    """
    Run asset-level spread backtest.

    Parameters
    ----------
    prices:
        Log price dataframe.

    weights:
        Spread portfolio weights.

    signal:
        Position signal.

            +1 : long spread
            -1 : short spread
             0 : flat

    initial_capital:
        Initial equity value.

    Returns
    -------
    BacktestResult
    """

    if not prices.index.equals(signal.index):
        raise ValueError("prices and signal index must match.")

    weights = np.asarray(
        weights,
        dtype=float,
    )

    if prices.shape[1] != len(weights):
        raise ValueError("Number of assets and weights must match.")

    # ------------------------------------------
    # Asset return
    # ------------------------------------------

    asset_returns = prices.diff().fillna(0)

    # ------------------------------------------
    # Spread return
    #
    # w' ΔX
    # ------------------------------------------

    spread_returns = asset_returns.dot(weights)

    # ------------------------------------------
    # Execute next bar
    # ------------------------------------------

    position = signal.shift(1).fillna(0)

    # ------------------------------------------
    # PnL
    # ------------------------------------------

    pnl = position * spread_returns

    cumulative_pnl = pnl.cumsum()

    equity = initial_capital + cumulative_pnl

    return BacktestResult(
        pnl=pnl,
        cumulative_pnl=cumulative_pnl,
        equity=equity,
    )
