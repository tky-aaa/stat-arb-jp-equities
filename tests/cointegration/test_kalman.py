import numpy as np
import pandas as pd

from statarb.cointegration.kalman import (
    kalman_spread,
)


def test_kalman_spread():

    prices = pd.DataFrame(
        {
            "A": [
                20.0,
                21.0,
                22.0,
                23.0,
                24.0,
            ],
            "B": [
                10.0,
                10.5,
                11.0,
                11.5,
                12.0,
            ],
            "C": [
                5.0,
                5.2,
                5.5,
                5.7,
                6.0,
            ],
        }
    )

    result = kalman_spread(
        prices,
        initial_beta=[
            -1.0,
            0.5,
        ],
    )

    assert len(result.spread) == len(prices)

    assert len(result.beta) == len(prices)

    assert list(result.beta.columns) == [
        "beta_B",
        "beta_C",
    ]

    assert result.intercept.name == "intercept"

    assert len(result.intercept) == len(prices)

    assert np.isfinite(
        result.spread,
    ).all()

    assert np.isfinite(
        result.beta.values,
    ).all()

    assert np.isfinite(
        result.intercept.values,
    ).all()
