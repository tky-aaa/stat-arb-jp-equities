import pandas as pd

from statarb.config.contract import KalmanSpread


def test_kalman_spread_contains_filtered_outputs() -> None:
    index = pd.date_range(
        "2025-01-01",
        periods=3,
        freq="D",
    )

    betas = pd.DataFrame(
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

    kalman_spread = KalmanSpread(
        tickers=["Y", "X1", "X2"],
        betas=betas,
        intercept=intercept,
        values=values,
    )

    assert kalman_spread.tickers == [
        "Y",
        "X1",
        "X2",
    ]

    pd.testing.assert_frame_equal(
        kalman_spread.betas,
        betas,
    )

    pd.testing.assert_series_equal(
        kalman_spread.intercept,
        intercept,
    )

    pd.testing.assert_series_equal(
        kalman_spread.values,
        values,
    )
