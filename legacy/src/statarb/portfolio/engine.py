from dataclasses import dataclass

import pandas as pd

from ..backtest.pipeline import run_spread_backtest
from ..cointegration.selection import SelectedSpread


@dataclass(frozen=True)
class PortfolioBacktestResult:
    """
    Result of multi-spread portfolio backtest.
    """

    returns: pd.Series

    equity: pd.Series

    spread_returns: pd.DataFrame

    weights_sum: float

    @property
    def pnl(self) -> pd.Series:
        """
        Alias for portfolio returns.
        """
        return self.returns

    @property
    def cumulative_pnl(self) -> pd.Series:
        """
        Alias for cumulative portfolio PnL.
        """
        return self.equity - 1

    @property
    def spread_pnls(self) -> pd.DataFrame:
        """
        Alias for individual spread returns.
        """
        return self.spread_returns


def combine_backtest_results(
    returns_list: list[pd.Series],
    weights: list[float],
) -> PortfolioBacktestResult:
    """
    Combine individual spread backtest returns.
    """

    if len(returns_list) != len(weights):
        raise ValueError("Number of returns and weights must match.")

    weighted_returns = {}

    for i, (
        returns,
        weight,
    ) in enumerate(
        zip(
            returns_list,
            weights,
        )
    ):
        weighted_returns[i] = returns * weight

    spread_returns = pd.DataFrame(
        weighted_returns,
    )

    portfolio_returns = spread_returns.sum(
        axis=1,
    )

    equity = (1 + portfolio_returns).cumprod()

    return PortfolioBacktestResult(
        returns=portfolio_returns,
        equity=equity,
        spread_returns=spread_returns,
        weights_sum=sum(weights),
    )


def run_portfolio_backtest(
    selected_spreads: list[SelectedSpread],
    log_prices: pd.DataFrame,
    weights: list[float],
    *,
    window: int = 60,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    threshold_method: str = "fixed",
) -> PortfolioBacktestResult:
    """
    Run portfolio backtest for multiple spreads.

    Pipeline:

        SelectedSpread
              |
              v
        spread backtest
              |
              v
        individual returns
              |
              v
        weighted portfolio
    """

    if len(selected_spreads) != len(weights):
        raise ValueError("Number of spreads and weights must match.")

    returns_list = []

    for selected in selected_spreads:
        result = run_spread_backtest(
            selected,
            log_prices,
            window=window,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            threshold_method=threshold_method,
        )

        returns_list.append(
            result.backtest.pnl,
        )

    return combine_backtest_results(
        returns_list,
        weights,
    )
