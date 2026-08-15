import numpy as np
import pandas as pd

from newstatarb.config.contract import Prices, SubGroup
from newstatarb.core.screening.api import ScreeningAPI


def _prices() -> Prices:
    index = pd.date_range(
        "2025-01-01",
        periods=20,
        freq="D",
    )

    training = pd.DataFrame(
        {
            "AAA": np.linspace(100, 120, 20),
            "BBB": np.linspace(100, 121, 20),
            "CCC": np.linspace(100, 130, 20),
            "DDD": np.linspace(100, 131, 20),
        },
        index=index,
    )

    test = training.copy()

    return Prices(
        training=training,
        test=test,
    )


def test_full_generates_subgroups_from_entire_universe() -> None:
    api = ScreeningAPI(
        screening_method="full",
        pca_components=2,
        n_clusters=2,
        min_cluster_size=2,
        max_cluster_size=4,
        min_assets=2,
        max_assets=2,
    )

    result = api.service(
        prices=_prices(),
        factors=pd.DataFrame(),
    )

    assert result == [
        SubGroup(tickers=["AAA", "BBB"]),
        SubGroup(tickers=["AAA", "CCC"]),
        SubGroup(tickers=["AAA", "DDD"]),
        SubGroup(tickers=["BBB", "CCC"]),
        SubGroup(tickers=["BBB", "DDD"]),
        SubGroup(tickers=["CCC", "DDD"]),
    ]
