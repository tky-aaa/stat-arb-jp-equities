import numpy as np
import pandas as pd
import pytest

from statarb.core.signal.kalman import KalmanFilter


@pytest.fixture
def prices() -> pd.DataFrame:
    index = pd.date_range(
        "2025-01-01",
        periods=100,
        freq="D",
    )
    x1 = np.linspace(100.0, 120.0, 100)
    x2 = 80.0 + 15.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, 100))
    y = 2.0 + 0.5 * x1 - 0.3 * x2
    return pd.DataFrame(
        {
            "Y": y,
            "X1": x1,
            "X2": x2,
        },
        index=index,
    )


def test_initialize_returns_expected_dimensions(
    prices: pd.DataFrame,
) -> None:
    kalman = KalmanFilter()
    (
        state,
        covariance,
        observation_variance,
        transition_covariance,
    ) = kalman.initialize(
        prices,
        alpha=1e-5,
    )

    assert state.shape == (3,)
    assert covariance.shape == (3, 3)
    assert transition_covariance.shape == (3, 3)
    assert np.isfinite(state).all()
    assert np.isfinite(covariance).all()
    assert np.isfinite(observation_variance)
    assert np.isfinite(transition_covariance).all()


def test_initialize_uses_least_squares_state(
    prices: pd.DataFrame,
) -> None:
    kalman = KalmanFilter()

    (
        state,
        _,
        _,
        _,
    ) = kalman.initialize(
        prices,
        alpha=1e-5,
    )

    expected = np.array(
        [
            2.0,
            0.5,
            -0.3,
        ]
    )

    np.testing.assert_allclose(
        state,
        expected,
        atol=1e-10,
    )


def test_filter_returns_one_step_ahead_spread(
    prices: pd.DataFrame,
) -> None:
    kalman = KalmanFilter()

    (
        state,
        covariance,
        observation_variance,
        transition_covariance,
    ) = kalman.initialize(
        prices,
        alpha=1e-5,
    )

    (
        spread,
        beta,
        intercept,
        final_state,
        final_covariance,
    ) = kalman.filter(
        prices,
        state=state,
        covariance=covariance,
        observation_variance=observation_variance,
        transition_covariance=transition_covariance,
    )

    assert len(spread) == len(prices)
    assert len(beta) == len(prices)
    assert len(intercept) == len(prices)

    assert beta.shape == (len(prices), 3)
    assert list(beta.columns) == [
        "Y",
        "X1",
        "X2",
    ]

    assert intercept.shape == (len(prices),)
    assert final_state.shape == (3,)
    assert final_covariance.shape == (3, 3)

    assert np.isfinite(spread).all()
    assert np.isfinite(beta.to_numpy()).all()
    assert np.isfinite(intercept.to_numpy()).all()


def test_filter_preserves_index(
    prices: pd.DataFrame,
) -> None:
    kalman = KalmanFilter()

    (
        state,
        covariance,
        observation_variance,
        transition_covariance,
    ) = kalman.initialize(
        prices,
        alpha=1e-5,
    )

    (
        spread,
        beta,
        intercept,
        _,
        _,
    ) = kalman.filter(
        prices,
        state=state,
        covariance=covariance,
        observation_variance=observation_variance,
        transition_covariance=transition_covariance,
    )

    pd.testing.assert_index_equal(
        spread.index,
        prices.index,
    )
    pd.testing.assert_index_equal(
        beta.index,
        prices.index,
    )
    pd.testing.assert_index_equal(
        intercept.index,
        prices.index,
    )


def test_initialize_rejects_invalid_alpha(
    prices: pd.DataFrame,
) -> None:
    kalman = KalmanFilter()

    with pytest.raises(ValueError):
        kalman.initialize(
            prices,
            alpha=0.0,
        )


def test_initialize_rejects_single_asset(
    prices: pd.DataFrame,
) -> None:
    kalman = KalmanFilter()

    with pytest.raises(ValueError):
        kalman.initialize(
            prices[["Y"]],
            alpha=1e-5,
        )


def test_filter_current_observation_does_not_affect_previous_spreads(
    prices: pd.DataFrame,
) -> None:
    kalman = KalmanFilter()

    (
        state,
        covariance,
        observation_variance,
        transition_covariance,
    ) = kalman.initialize(
        prices,
        alpha=1e-5,
    )

    (
        original_spread,
        _,
        _,
        _,
        _,
    ) = kalman.filter(
        prices,
        state=state,
        covariance=covariance,
        observation_variance=observation_variance,
        transition_covariance=transition_covariance,
    )

    modified_prices = prices.copy()
    modified_prices.iloc[50, 0] += 100.0

    (
        modified_spread,
        _,
        _,
        _,
        _,
    ) = kalman.filter(
        modified_prices,
        state=state,
        covariance=covariance,
        observation_variance=observation_variance,
        transition_covariance=transition_covariance,
    )

    pd.testing.assert_series_equal(
        original_spread.iloc[:50],
        modified_spread.iloc[:50],
    )


def test_filter_supports_multiple_explanatory_assets(
    prices: pd.DataFrame,
) -> None:
    kalman = KalmanFilter()

    x3 = 60.0 + 8.0 * np.cos(
        np.linspace(
            0.0,
            2.0 * np.pi,
            len(prices),
        )
    )

    prices = prices.copy()
    prices["X3"] = x3

    (
        state,
        covariance,
        observation_variance,
        transition_covariance,
    ) = kalman.initialize(
        prices,
        alpha=1e-5,
    )

    assert state.shape == (4,)
    assert covariance.shape == (4, 4)
    assert transition_covariance.shape == (4, 4)

    (
        spread,
        beta,
        intercept,
        final_state,
        final_covariance,
    ) = kalman.filter(
        prices,
        state=state,
        covariance=covariance,
        observation_variance=observation_variance,
        transition_covariance=transition_covariance,
    )

    assert len(spread) == len(prices)

    assert beta.shape == (len(prices), 4)
    assert list(beta.columns) == [
        "Y",
        "X1",
        "X2",
        "X3",
    ]

    assert intercept.shape == (len(prices),)
    assert final_state.shape == (4,)
    assert final_covariance.shape == (4, 4)

    assert np.isfinite(spread).all()
    assert np.isfinite(beta.to_numpy()).all()
    assert np.isfinite(intercept.to_numpy()).all()


def test_filter_returns_spread_beta_and_intercept(
    prices: pd.DataFrame,
) -> None:
    kalman = KalmanFilter()

    (
        state,
        covariance,
        observation_variance,
        transition_covariance,
    ) = kalman.initialize(
        prices,
        alpha=1e-5,
    )

    (
        spread,
        betas,
        intercept,
        _,
        _,
    ) = kalman.filter(
        prices,
        state=state,
        covariance=covariance,
        observation_variance=observation_variance,
        transition_covariance=transition_covariance,
    )

    assert isinstance(spread, pd.Series)
    assert isinstance(betas, pd.DataFrame)
    assert isinstance(intercept, pd.Series)

    assert betas.shape == (len(prices), 3)
    assert list(betas.columns) == [
        "Y",
        "X1",
        "X2",
    ]

    assert intercept.shape == (len(prices),)

    pd.testing.assert_index_equal(
        spread.index,
        prices.index,
    )
    pd.testing.assert_index_equal(
        betas.index,
        prices.index,
    )
    pd.testing.assert_index_equal(
        intercept.index,
        prices.index,
    )
