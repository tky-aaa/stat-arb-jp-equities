import numpy as np
import pandas as pd

from statarb.config.contract import Prices, SubGroup
from statarb.core.cointegration.api import CointegrationAPI


def test_service_generates_cointegration_analyses() -> None:
    rng = np.random.default_rng(42)

    n = 120

    common = np.cumsum(
        rng.normal(0.0, 0.01, n),
    )

    spread = np.zeros(n)

    for i in range(1, n):
        spread[i] = 0.5 * spread[i - 1] + rng.normal(0.0, 0.002)

    log_prices = pd.DataFrame(
        {
            "AAA": common,
            "BBB": common + spread,
        },
        index=pd.date_range(
            "2025-01-01",
            periods=n,
            freq="D",
        ),
    )

    prices = Prices(
        training=np.exp(log_prices),
        test=pd.DataFrame(),
    )

    subgroup = SubGroup(
        tickers=["AAA", "BBB"],
    )

    api = CointegrationAPI(
        persistence_days=30,
    )

    analyses = api.service(
        prices=prices,
        subgroups=[subgroup],
    )

    assert isinstance(analyses, list)

    if analyses:
        analysis = analyses[0]

        assert analysis.spread.tickers == ["AAA", "BBB"]
        assert analysis.rank >= 1
        assert analysis.beta_index >= 0
        assert analysis.spread.tickers == ["AAA", "BBB"]


def test_service_skips_subgroup_with_too_few_common_observations() -> None:
    dates = pd.date_range("2025-01-01", periods=40, freq="D")

    prices = Prices(
        training=pd.DataFrame(
            {
                "A": range(100, 140),
                "B": range(200, 240),
                "C": [float("nan")] * 15 + list(range(300, 325)),
            },
            index=dates,
        ),
        test=pd.DataFrame(),
    )

    subgroups = [
        SubGroup(tickers=["A", "B", "C"]),
    ]

    api = CointegrationAPI(persistence_days=10)

    analyses = api.service(prices, subgroups)

    assert analyses == []
