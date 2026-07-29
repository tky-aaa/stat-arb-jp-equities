import numpy as np
import pandas as pd

from statarb.cointegration.evaluate_all import (
    evaluate_all_candidates,
)


def test_evaluate_all_candidates():

    np.random.seed(0)

    n = 200

    dates = pd.date_range(
        "2024-01-01",
        periods=n,
    )

    # ==========================
    # Common stochastic trend
    # ==========================

    base = np.cumsum(
        np.random.normal(
            0,
            0.01,
            n,
        )
    )

    # ==========================
    # Mean reverting spread
    # ==========================

    spread = np.zeros(n)

    for t in range(1, n):
        spread[t] = 0.7 * spread[t - 1] + np.random.normal(
            0,
            0.01,
        )

    # ==========================
    # Cointegrated prices
    # ==========================

    log_prices = pd.DataFrame(
        {
            "A": base + spread / 2,
            "B": base - spread / 2,
        },
        index=dates,
    )

    johansen_results = pd.DataFrame(
        {
            "tickers": [
                [
                    "A",
                    "B",
                ],
            ],
            "rank": [
                1,
            ],
            "beta": [
                [
                    [1.0],
                    [-1.0],
                ],
            ],
            "error": [
                None,
            ],
        }
    )

    result = evaluate_all_candidates(
        johansen_results,
        log_prices,
        persistence_window=100,
        persistence_maxlags=5,
    )

    assert len(result) == 1

    expected_columns = [
        "tickers",
        "rank",
        "beta_index",
        "beta",
        "adf_stat",
        "adf_pvalue",
        "kpss_stat",
        "kpss_pvalue",
        "rho1",
        "phi",
        "half_life",
        "mean",
        "variance",
        "std",
        "portmanteau",
        "persistence",
    ]

    for column in expected_columns:
        assert column in result.columns

    assert "error" not in result.columns

    assert result["persistence"].notna().all()

    numeric_columns = [
        "adf_stat",
        "adf_pvalue",
        "kpss_stat",
        "kpss_pvalue",
        "rho1",
        "phi",
        "half_life",
        "mean",
        "variance",
        "std",
        "portmanteau",
        "persistence",
    ]

    for column in numeric_columns:
        assert np.isfinite(result[column]).all()
