import pandas as pd

from .johansen import estimate_cointegration


def rolling_cointegration_persistence(
    log_prices: pd.DataFrame,
    tickers: list[str],
    *,
    window: int = 120,
    maxlags: int = 10,
) -> float:
    """
    Estimate rolling Johansen cointegration persistence.

    Parameters
    ----------
    log_prices:
        Log price dataframe.

    tickers:
        Assets in one candidate.

    window:
        Rolling window length.

    Returns
    -------
    float
        Fraction of windows with rank > 0.
    """

    prices = log_prices[tickers].dropna()

    if len(prices) < window:
        raise ValueError("Too few observations for rolling persistence.")

    total = 0
    success = 0

    for start in range(
        len(prices) - window + 1,
    ):
        sample = prices.iloc[start : start + window]

        result = estimate_cointegration(
            sample,
            maxlags=maxlags,
        )

        total += 1

        if result.rank > 0:
            success += 1

    return success / total
