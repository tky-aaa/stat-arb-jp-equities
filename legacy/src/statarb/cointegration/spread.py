import numpy as np
import pandas as pd


def create_spread(
    prices: pd.DataFrame,
    beta,
) -> pd.Series:
    """
    Create cointegration spread.

    spread = beta' x_t

    Parameters
    ----------
    prices:
        Price dataframe.

    beta:
        Cointegration vector.

    Returns
    -------
    pd.Series
        Spread time series.
    """

    beta = np.asarray(
        beta,
        dtype=float,
    ).flatten()

    if prices.shape[1] != len(beta):
        raise ValueError("Price dimension and beta dimension mismatch.")

    spread = prices.values @ beta

    return pd.Series(
        spread,
        index=prices.index,
        name="spread",
    )


def create_spreads(
    prices: pd.DataFrame,
    beta,
) -> pd.DataFrame:

    beta = np.asarray(
        beta,
        dtype=float,
    )

    if beta.ndim == 1:
        beta = beta.reshape(-1, 1)

    if prices.shape[1] != beta.shape[0]:
        raise ValueError("Price dimension and beta dimension mismatch.")

    spreads = prices.values @ beta

    columns = [f"spread_{i + 1}" for i in range(beta.shape[1])]

    return pd.DataFrame(
        spreads,
        index=prices.index,
        columns=columns,
    )
