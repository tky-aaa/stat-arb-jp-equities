import numpy as np
import pandas as pd


class KalmanFilter:
    def initialize(
        self,
        prices: pd.DataFrame,
        *,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Initialize a Kalman filter using the Palomar heuristic.

        Model
        -----
        y_t = mu_t + beta_t' x_t + epsilon_t

        theta_t = [mu_t, beta_t']

        Parameters
        ----------
        prices:
            Training price data. The first column is the dependent
            variable and the remaining columns are explanatory variables.

        alpha:
            Hyperparameter controlling the variability of the
            time-varying state.

        Returns
        -------
        state:
            Initial state vector [mu_LS, beta_LS].

        covariance:
            Initial state covariance matrix.

        observation_variance:
            Observation noise variance.

        transition_covariance:
            State transition noise covariance.
        """

        if prices.shape[1] < 2:
            raise ValueError("At least two assets are required.")

        if alpha <= 0:
            raise ValueError("alpha must be positive.")

        prices = prices.dropna()

        if prices.empty:
            raise ValueError("No valid observations.")

        y = prices.iloc[:, 0]
        X = prices.iloc[:, 1:]

        n_assets = X.shape[1]
        n_obs = len(prices)

        design = np.column_stack(
            [
                np.ones(n_obs),
                X.to_numpy(),
            ]
        )

        state = np.linalg.lstsq(
            design,
            y.to_numpy(),
            rcond=None,
        )[0]

        residuals = y.to_numpy() - design @ state

        observation_variance = float(
            np.var(
                residuals,
                ddof=1,
            )
        )

        if not np.isfinite(observation_variance):
            raise ValueError("Observation variance must be finite.")

        x_variance = X.var(
            ddof=1,
        ).to_numpy()

        if np.any(x_variance <= 0):
            raise ValueError("Explanatory variables must have positive variance.")

        initial_variance = np.empty(
            n_assets + 1,
            dtype=float,
        )

        initial_variance[0] = observation_variance / n_obs

        initial_variance[1:] = observation_variance / (n_obs * x_variance)

        covariance = np.diag(initial_variance)

        transition_variance = np.empty(
            n_assets + 1,
            dtype=float,
        )

        transition_variance[0] = alpha * observation_variance

        transition_variance[1:] = alpha * observation_variance / x_variance

        transition_covariance = np.diag(transition_variance)

        return (
            state,
            covariance,
            observation_variance,
            transition_covariance,
        )

    def filter(
        self,
        prices: pd.DataFrame,
        *,
        state: np.ndarray,
        covariance: np.ndarray,
        observation_variance: float,
        transition_covariance: np.ndarray,
    ) -> tuple[pd.Series, pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
        """
        Apply Kalman filtering to price observations.

        Returns
        -------
        spread:
            One-step-ahead prediction errors.

        beta:
            Filtered time-varying hedge ratios.

        intercept:
            Filtered time-varying intercept.
        """

        if prices.shape[1] < 2:
            raise ValueError("At least two assets are required.")

        prices = prices.dropna().astype(float)

        if prices.empty:
            raise ValueError("No valid observations.")

        y = prices.iloc[:, 0]
        X = prices.iloc[:, 1:]

        n_assets = X.shape[1]
        state_dimension = n_assets + 1

        if len(state) != state_dimension:
            raise ValueError("State dimension mismatch.")

        if covariance.shape != (
            state_dimension,
            state_dimension,
        ):
            raise ValueError("Covariance dimension mismatch.")

        if transition_covariance.shape != (
            state_dimension,
            state_dimension,
        ):
            raise ValueError("Transition covariance dimension mismatch.")

        state = np.asarray(
            state,
            dtype=float,
        ).copy()

        covariance = np.asarray(
            covariance,
            dtype=float,
        ).copy()

        betas = []
        intercepts = []
        spreads = []

        for t in range(len(prices)):
            observation = np.concatenate(
                [
                    np.array([1.0]),
                    X.iloc[t].to_numpy(),
                ]
            )

            # Prediction
            state_prior = state.copy()

            covariance_prior = covariance + transition_covariance

            # One-step-ahead prediction error.
            #
            # y_t is not used to update the state
            # until after the spread is calculated.
            prediction = observation @ state_prior

            spread = y.iloc[t] - prediction

            prediction_variance = (
                observation @ covariance_prior @ observation + observation_variance
            )

            if prediction_variance <= 0:
                raise ValueError("Prediction variance must be positive.")

            gain = covariance_prior @ observation / prediction_variance

            # Update
            error = y.iloc[t] - prediction

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

            spreads.append(spread)

        beta_columns = [f"beta_{ticker}" for ticker in X.columns]

        beta = pd.DataFrame(
            betas,
            index=prices.index,
            columns=beta_columns,
        )

        intercept = pd.Series(
            intercepts,
            index=prices.index,
            name="intercept",
        )

        spread = pd.Series(
            spreads,
            index=prices.index,
            name="spread",
        )

        return (
            spread,
            beta,
            intercept,
            state,
            covariance,
        )
