import numpy as np
import pandas as pd

from .evaluation import evaluate_spread
from .persistence import rolling_spread_persistence
from .spread import create_spreads


def evaluate_all_candidates(
    johansen_results: pd.DataFrame,
    log_prices: pd.DataFrame,
    *,
    persistence_window: int = 120,
) -> pd.DataFrame:

    rows = []

    for _, result in johansen_results.iterrows():
        if result["rank"] <= 0:
            continue

        tickers = result["tickers"]

        prices = log_prices[tickers]

        beta = np.asarray(
            result["beta"],
            dtype=float,
        )

        spreads = create_spreads(
            prices,
            beta,
        )

        for i, column in enumerate(
            spreads.columns,
        ):
            spread = spreads[column]

            evaluation = evaluate_spread(
                spread,
            )

            persistence = rolling_spread_persistence(
                spread,
                window=persistence_window,
            )

            rows.append(
                {
                    "tickers": tickers,
                    "rank": result["rank"],
                    "beta_index": i,
                    "beta": beta[:, i],
                    "spread": column,
                    "persistence": persistence,
                    **evaluation.__dict__,
                }
            )

    return pd.DataFrame(rows)
