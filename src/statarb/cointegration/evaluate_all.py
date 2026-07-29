import pandas as pd

from .evaluation_pipeline import evaluate_candidates
from .persistence import rolling_cointegration_persistence


def evaluate_all_candidates(
    johansen_results: pd.DataFrame,
    log_prices: pd.DataFrame,
    *,
    persistence_window: int = 120,
    persistence_maxlags: int = 10,
) -> pd.DataFrame:

    evaluation_results = evaluate_candidates(
        johansen_results,
        log_prices,
    )

    if len(evaluation_results) == 0:
        return evaluation_results

    evaluation_results["tickers"] = evaluation_results["tickers"].apply(tuple)

    persistence_values = []

    for _, row in evaluation_results.iterrows():
        tickers = list(row["tickers"])

        persistence = rolling_cointegration_persistence(
            log_prices,
            tickers,
            window=persistence_window,
            maxlags=persistence_maxlags,
        )

        persistence_values.append(persistence)

    evaluation_results["persistence"] = persistence_values

    return evaluation_results
