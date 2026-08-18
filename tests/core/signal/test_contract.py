import pandas as pd

from statarb.config.contract import Spread


def test_spread_contains_filtered_outputs() -> None:

    index = pd.date_range(
        "2025-01-01",
        periods=3,
        freq="D",
    )

    beta = pd.DataFrame(
        {
            "beta_X1": [0.5, 0.51, 0.52],
            "beta_X2": [-0.3, -0.31, -0.32],
        },
        index=index,
    )

    intercept = pd.Series(
        [2.0, 2.1, 2.2],
        index=index,
        name="intercept",
    )

    values = pd.Series(
        [0.1, -0.2, 0.3],
        index=index,
        name="spread",
    )

    spread = Spread(
        tickers=["Y", "X1", "X2"],
        beta=beta,
        intercept=intercept,
        values=values,
    )

    assert spread.tickers == [
        "Y",
        "X1",
        "X2",
    ]

    pd.testing.assert_frame_equal(
        spread.beta,
        beta,
    )

    pd.testing.assert_series_equal(
        spread.intercept,
        intercept,
    )

    pd.testing.assert_series_equal(
        spread.values,
        values,
    )
