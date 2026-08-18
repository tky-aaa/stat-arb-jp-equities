import numpy as np
import pandas as pd
import pytest

from statarb.core.cointegration.spread_creator import SpreadCreator


def test_create_returns_expected_spread() -> None:
    prices = pd.DataFrame(
        {
            "AAA": [1.0, 2.0, 3.0],
            "BBB": [4.0, 5.0, 6.0],
        }
    )

    beta = np.array([1.0, -0.5])

    creator = SpreadCreator()

    spread = creator.create(
        prices=prices,
        tickers=["AAA", "BBB"],
        beta=beta,
    )

    expected = pd.Series(
        [-1.0, -0.5, 0.0],
        index=prices.index,
        name="spread",
    )

    pd.testing.assert_series_equal(
        spread.values,
        expected,
    )

    assert spread.tickers == ["AAA", "BBB"]

    expected_beta = np.tile(
        beta,
        (len(prices), 1),
    )

    np.testing.assert_array_equal(
        spread.beta,
        expected_beta,
    )


def test_create_rejects_price_beta_dimension_mismatch() -> None:
    prices = pd.DataFrame(
        {
            "AAA": [1.0, 2.0],
            "BBB": [3.0, 4.0],
        }
    )

    creator = SpreadCreator()

    with pytest.raises(ValueError):
        creator.create(
            prices=prices,
            tickers=["AAA", "BBB"],
            beta=np.array([1.0]),
        )


def test_create_rejects_ticker_beta_dimension_mismatch() -> None:
    prices = pd.DataFrame(
        {
            "AAA": [1.0, 2.0],
            "BBB": [3.0, 4.0],
        }
    )

    creator = SpreadCreator()

    with pytest.raises(ValueError):
        creator.create(
            prices=prices,
            tickers=["AAA"],
            beta=np.array([1.0, -0.5]),
        )
