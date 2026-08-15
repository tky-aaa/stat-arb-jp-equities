import numpy as np
import pandas as pd


def _to_log_price(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert prices to log-prices.

    Parameters
    ----------
    prices : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive.")

    return np.log(prices)


def preprocess_prices(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standard preprocessing pipeline.
    """
    prices = _to_log_price(prices)

    return prices


def compute_log_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute log returns from prices.

    Parameters
    ----------
    prices:
        Price series.

    Returns
    -------
    pd.DataFrame
        Log return series.
    """

    return preprocess_prices(prices).diff().dropna()
