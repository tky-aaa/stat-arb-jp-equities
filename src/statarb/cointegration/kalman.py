from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KalmanResult:
    """
    Result of Kalman filter hedge ratio estimation.
    """

    spread: pd.Series

    beta: pd.DataFrame

    intercept: pd.Series


def kalman_spread(
    prices: pd.DataFrame,
    initial_beta: list[float] | None = None,
    *,
    delta: float = 1e-5,
    observation_variance: float = 1e-3,
) -> KalmanResult:
    """
    Estimate time-varying hedge ratio using Kalman filter.

    Model:

        y_t = mu_t + beta_t' x_t + eps_t

    State:

        theta_t =
        [
            mu_t,
            beta_t
        ]

    State dynamics:

        theta_t = theta_{t-1} + noise
    """

    if prices.shape[1] < 2:
        raise ValueError("At least two assets are required.")

    prices = prices.dropna()

    y = prices.iloc[:, 0]

    X = prices.iloc[:, 1:]

    explanatory_tickers = list(X.columns)

    n_assets = len(explanatory_tickers)

    if initial_beta is None:
        initial_beta = [0.0 for _ in range(n_assets)]

    if len(initial_beta) != n_assets:
        raise ValueError("initial_beta dimension mismatch.")

    state_dimension = n_assets + 1

    state = np.zeros(
        state_dimension,
    )

    state[1:] = np.asarray(
        initial_beta,
        dtype=float,
    )

    covariance = np.eye(
        state_dimension,
    )

    transition_covariance = (
        delta
        / (1 - delta)
        * np.eye(
            state_dimension,
        )
    )

    betas = []

    intercepts = []

    spreads = []

    for t in range(
        len(prices),
    ):
        observation = np.concatenate(
            [
                np.array([1.0]),
                X.iloc[t].values,
            ]
        )

        # Prediction step

        state_prior = state.copy()

        covariance_prior = covariance + transition_covariance

        prediction = observation @ state_prior

        error = y.iloc[t] - prediction

        variance = observation @ covariance_prior @ observation + observation_variance

        gain = covariance_prior @ observation / variance

        # Update step

        state = state_prior + gain * error

        covariance = (
            covariance_prior
            - np.outer(
                gain,
                observation,
            )
            @ covariance_prior
        )

        covariance = (covariance + covariance.T) / 2

        intercepts.append(state[0])

        betas.append(state[1:].copy())

        spread_value = y.iloc[t] - state[0] - state[1:] @ X.iloc[t].values

        spreads.append(spread_value)

    beta_columns = [f"beta_{ticker}" for ticker in explanatory_tickers]

    beta_df = pd.DataFrame(
        betas,
        columns=beta_columns,
        index=prices.index,
    )

    intercept_series = pd.Series(
        intercepts,
        index=prices.index,
        name="intercept",
    )

    spread_series = pd.Series(
        spreads,
        index=prices.index,
        name="spread",
    )

    return KalmanResult(
        spread=spread_series,
        beta=beta_df,
        intercept=intercept_series,
    )
