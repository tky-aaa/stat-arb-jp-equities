import numpy as np
import pandas as pd

from statarb.cointegration.evaluation_pipeline import evaluation_pipeline
from statarb.screening.candidate import CandidateGroup


def test_evaluation_pipeline_runs():

    np.random.seed(0)

    n = 200

    dates = pd.date_range(
        "2024-01-01",
        periods=n,
        freq="B",
    )

    base = np.cumsum(
        np.random.normal(
            0,
            0.01,
            n,
        )
    )

    spread = np.zeros(n)

    for t in range(1, n):
        spread[t] = 0.8 * spread[t - 1] + np.random.normal(
            0,
            0.01,
        )

    log_prices = pd.DataFrame(
        {
            "A": base,
            "B": base + spread,
            "C": base - spread,
            "D": np.cumsum(
                np.random.normal(
                    0,
                    0.02,
                    n,
                )
            ),
        },
        index=dates,
    )

    group = CandidateGroup(
        tickers=[
            "A",
            "B",
            "C",
            "D",
        ]
    )

    result = evaluation_pipeline(
        group,
        log_prices,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert len(result) >= 1

    expected_columns = [
        "tickers",
        "spread",
        "rank",
        "persistence",
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
    ]

    for column in expected_columns:
        assert column in result.columns
