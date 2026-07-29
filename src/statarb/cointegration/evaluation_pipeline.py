import numpy as np
import pandas as pd

from .evaluation import evaluate_spread
from .spread import create_spread


def evaluate_candidates(
    johansen_results: pd.DataFrame,
    log_prices: pd.DataFrame,
    *,
    max_lag: int = 10,
) -> pd.DataFrame:
    """
    Evaluate Johansen cointegration vectors.

    One Johansen result may contain
    multiple cointegration vectors.

    Each beta vector is evaluated
    as an independent spread.
    """

    records = []

    candidates = johansen_results[
        (johansen_results["error"].isna()) & (johansen_results["rank"] > 0)
    ]

    for _, row in candidates.iterrows():
        tickers = list(row["tickers"])

        beta_matrix = np.asarray(
            row["beta"],
            dtype=float,
        )

        rank = int(row["rank"])

        if beta_matrix.shape != (
            len(tickers),
            rank,
        ):
            raise ValueError(
                f"Beta matrix dimension mismatch: "
                f"expected {(len(tickers), rank)}, "
                f"got {beta_matrix.shape}"
            )

        prices = log_prices[tickers]

        for beta_index in range(rank):
            beta = beta_matrix[:, beta_index]

            spread = create_spread(
                prices,
                beta,
            )

            evaluation = evaluate_spread(
                spread,
                max_lag=max_lag,
            )

            records.append(
                {
                    "tickers": tuple(tickers),
                    "rank": rank,
                    "beta_index": beta_index,
                    "beta": beta.tolist(),
                    "adf_stat": evaluation.adf_stat,
                    "adf_pvalue": evaluation.adf_pvalue,
                    "kpss_stat": evaluation.kpss_stat,
                    "kpss_pvalue": evaluation.kpss_pvalue,
                    "rho1": evaluation.rho1,
                    "phi": evaluation.phi,
                    "half_life": evaluation.half_life,
                    "mean": evaluation.mean,
                    "variance": evaluation.variance,
                    "std": evaluation.std,
                    "portmanteau": evaluation.portmanteau,
                }
            )

    return pd.DataFrame(records)
