from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..cointegration.kalman import (
    kalman_spread,
)
from ..cointegration.selection import (
    SelectedSpread,
)
from ..cointegration.spread import (
    create_spread,
)
from ..signals.optimizer import (
    select_empirical_threshold,
    select_gaussian_threshold,
)
from ..signals.strategy import (
    generate_threshold_signal,
)
from ..signals.zscore import (
    calculate_zscore,
)
from .engine import (
    BacktestResult,
    run_backtest,
)


@dataclass(frozen=True)
class SpreadBacktestResult:
    """
    Backtest result for one spread.
    """

    spread: pd.Series

    zscore: pd.Series

    signal: pd.Series

    backtest: BacktestResult


def run_spread_backtest(
    selected_spread: SelectedSpread,
    log_prices: pd.DataFrame,
    *,
    window: int = 60,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    threshold_method: str = "fixed",
    spread_method: str = "static",
) -> SpreadBacktestResult:
    """
    Run backtest for one spread candidate.
    """

    prices = log_prices[selected_spread.tickers]

    # ==================================================
    # Spread construction
    # ==================================================

    if spread_method == "static":
        spread = create_spread(
            prices,
            np.asarray(
                selected_spread.beta,
                dtype=float,
            ),
        )

    elif spread_method == "kalman":
        initial_beta = np.asarray(
            selected_spread.beta[1:],
            dtype=float,
        )

        kalman_result = kalman_spread(
            prices,
            initial_beta=initial_beta,
        )

        spread = kalman_result.spread

    else:
        raise ValueError("spread_method must be 'static' or 'kalman'.")

    # ==================================================
    # Z-score
    # ==================================================

    zscore = calculate_zscore(
        spread,
        window=window,
    )

    # ==================================================
    # Threshold selection
    # ==================================================

    if threshold_method == "fixed":
        selected_entry_threshold = entry_threshold

    elif threshold_method == "gaussian":
        threshold_result = select_gaussian_threshold()

        selected_entry_threshold = threshold_result.threshold

    elif threshold_method == "empirical":
        threshold_result = select_empirical_threshold(
            zscore,
        )

        selected_entry_threshold = threshold_result.threshold

    else:
        raise ValueError(
            "threshold_method must be 'fixed', 'gaussian', or 'empirical'."
        )

    # ==================================================
    # Signal
    # ==================================================

    signal = generate_threshold_signal(
        zscore,
        entry_threshold=selected_entry_threshold,
        exit_threshold=exit_threshold,
    )

    # ==================================================
    # Backtest
    # ==================================================

    backtest_result = run_backtest(
        prices,
        np.asarray(
            selected_spread.beta,
            dtype=float,
        ),
        signal,
    )

    return SpreadBacktestResult(
        spread=spread,
        zscore=zscore,
        signal=signal,
        backtest=backtest_result,
    )


def run_strategy_backtest(
    selected_spread: SelectedSpread,
    log_prices: pd.DataFrame,
    *,
    window: int = 60,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    threshold_method: str = "fixed",
    spread_method: str = "static",
) -> SpreadBacktestResult:
    """
    Backward compatible wrapper.

    Deprecated:
        Use run_spread_backtest().
    """

    return run_spread_backtest(
        selected_spread,
        log_prices,
        window=window,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        threshold_method=threshold_method,
        spread_method=spread_method,
    )
