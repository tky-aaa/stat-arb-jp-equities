import numpy as np
import pandas as pd

from statarb.cointegration.search import (
    search_cointegrated_subgroups,
)
from statarb.screening.candidate import CandidateGroup


def test_search_runs():

    np.random.seed(0)

    dates = pd.date_range(
        "2024-01-01",
        periods=300,
        freq="B",
    )

    base = np.cumsum(
        np.random.normal(
            0,
            0.01,
            300,
        )
    )

    spread = np.zeros(300)

    for t in range(1, 300):
        spread[t] = 0.8 * spread[t - 1] + np.random.normal(
            0,
            0.01,
        )

    log_prices = pd.DataFrame(
        {
            "A": base,
            "B": base + spread,
            "C": base - spread,
            "D": np.cumsum(np.random.normal(0, 0.02, 300)),
        },
        index=dates,
    )

    group = CandidateGroup(tickers=["A", "B", "C", "D"])

    results = search_cointegrated_subgroups(
        group,
        log_prices,
    )

    assert isinstance(
        results,
        list,
    )
