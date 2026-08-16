import pandas as pd
import pytest

from statarb.core.backtest.portfolio_return_calculator import (
    PortfolioReturnCalculator,
)


def test_calculate_weighted_portfolio_returns() -> None:
    index = pd.date_range(
        "2025-01-01",
        periods=3,
    )

    returns = [
        pd.Series(
            [0.01, 0.02, -0.01],
            index=index,
        ),
        pd.Series(
            [0.03, -0.01, 0.02],
            index=index,
        ),
    ]

    weights = [0.6, 0.4]

    result = PortfolioReturnCalculator().calculate(
        returns,
        weights,
    )

    expected = pd.Series(
        [
            0.018,
            0.008,
            0.002,
        ],
        index=index,
        name="portfolio_return",
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_calculate_rejects_mismatched_returns_and_weights() -> None:
    returns = [
        pd.Series([0.01, 0.02]),
        pd.Series([0.03, 0.04]),
    ]

    weights = [0.5]

    with pytest.raises(
        ValueError,
        match="Number of returns and weights must match",
    ):
        PortfolioReturnCalculator().calculate(
            returns,
            weights,
        )


def test_calculate_empty_input() -> None:
    result = PortfolioReturnCalculator().calculate(
        [],
        [],
    )

    expected = pd.Series(
        dtype=float,
        name="portfolio_return",
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_calculate_aligns_returns_by_index() -> None:
    index_1 = pd.date_range(
        "2025-01-01",
        periods=3,
    )

    index_2 = pd.date_range(
        "2025-01-02",
        periods=3,
    )

    returns = [
        pd.Series(
            [0.01, 0.02, 0.03],
            index=index_1,
        ),
        pd.Series(
            [0.04, 0.05, 0.06],
            index=index_2,
        ),
    ]

    result = PortfolioReturnCalculator().calculate(
        returns,
        [0.5, 0.5],
    )

    assert result.index.equals(
        index_1.union(index_2),
    )

    assert result.loc["2025-01-01"] == pytest.approx(0.005)
    assert result.loc["2025-01-02"] == pytest.approx(0.03)
    assert result.loc["2025-01-03"] == pytest.approx(0.04)
    assert result.loc["2025-01-04"] == pytest.approx(0.03)


def test_calculate_preserves_portfolio_return_name() -> None:
    returns = [
        pd.Series(
            [0.01, 0.02],
        ),
    ]

    result = PortfolioReturnCalculator().calculate(
        returns,
        [1.0],
    )

    assert result.name == "portfolio_return"
