from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..cointegration.selection import SelectedSpread
from ..cointegration.spread import create_spread


@dataclass(frozen=True)
class SpreadPortfolio:
    """
    Portfolio allocation for spreads.
    """

    spreads: list[SelectedSpread]

    weights: list[float]


def _estimate_spread_volatility(
    spread: pd.Series,
) -> float:
    """
    Estimate spread volatility.
    """

    return float(spread.std())


def allocate_inverse_volatility(
    selected_spreads: list[SelectedSpread],
    log_prices: pd.DataFrame,
) -> SpreadPortfolio:
    """
    Allocate portfolio weights by inverse volatility.
    """

    volatilities = []

    for selected in selected_spreads:
        prices = log_prices[selected.tickers]

        spread = create_spread(
            prices,
            np.array(
                selected.beta,
            ),
        )

        volatility = _estimate_spread_volatility(
            spread,
        )

        if volatility <= 0:
            raise ValueError("Spread volatility must be positive.")

        volatilities.append(
            volatility,
        )

    inverse_vol = 1 / np.array(volatilities)

    weights = inverse_vol / inverse_vol.sum()

    return SpreadPortfolio(
        spreads=selected_spreads,
        weights=weights.tolist(),
    )
