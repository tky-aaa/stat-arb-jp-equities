import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from statarb.config.contract import Spread


class MeanReversionEvaluator:
    def _lag_autocorrelation(
        self,
        spread: pd.Series,
        lag: int,
    ) -> float:
        x = spread.dropna()

        if len(x) <= lag:
            return np.nan

        x = x - x.mean()

        numerator = (x.iloc[:-lag] * x.iloc[lag:]).mean()

        denominator = (x**2).mean()

        if denominator == 0:
            return np.nan

        return float(numerator / denominator)

    def _estimate_ar1_phi(
        self,
        spread: pd.Series,
    ) -> float:
        x = spread.dropna()

        lag = x.shift(1)

        df = pd.concat(
            [
                x,
                lag,
            ],
            axis=1,
        ).dropna()

        if len(df) < 2:
            return np.nan

        phi = np.polyfit(
            df.iloc[:, 1],
            df.iloc[:, 0],
            1,
        )[0]

        return float(phi)

    def _estimate_half_life(
        self,
        phi: float,
    ) -> float:
        if phi <= 0 or phi >= 1:
            return np.inf

        return float(-np.log(2) / np.log(phi))

    def _portmanteau(
        self,
        spread: pd.Series,
        max_lag: int,
    ) -> float:
        value = 0.0

        for lag in range(
            1,
            max_lag + 1,
        ):
            rho = self._lag_autocorrelation(
                spread,
                lag,
            )

            if np.isnan(rho):
                return np.nan

            value += rho**2

        return float(value)

    def _rolling_persistence(
        self,
        spread: pd.Series,
        window: int,
    ) -> float:
        if window <= 1:
            raise ValueError("window must be greater than 1.")

        values = spread.dropna().astype(float)

        if len(values) < window:
            return np.nan

        stationary_windows = 0
        total_windows = 0

        for end in range(
            window,
            len(values) + 1,
        ):
            window_values = values.iloc[end - window : end]

            try:
                adf_result = adfuller(
                    window_values,
                    autolag="AIC",
                )

                adf_pvalue = adf_result[1]

                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore",
                        InterpolationWarning,
                    )

                    kpss_result = kpss(
                        window_values,
                        regression="c",
                        nlags="auto",
                    )

                kpss_pvalue = kpss_result[1]

            except ValueError:
                continue

            total_windows += 1

            if adf_pvalue < 0.05 and kpss_pvalue > 0.05:
                stationary_windows += 1

        if total_windows == 0:
            return np.nan

        return float(stationary_windows / total_windows)

    def evaluate(
        self,
        spread: Spread,
        persistence_window: int,
        max_lag: int = 10,
    ) -> dict[str, float]:

        if persistence_window <= 1:
            raise ValueError("window must be greater than 1.")

        values = spread.values.dropna().astype(float)

        if len(values) < 50:
            raise ValueError("Too few observations.")

        rho1 = self._lag_autocorrelation(
            values,
            1,
        )

        phi = self._estimate_ar1_phi(
            values,
        )

        half_life = self._estimate_half_life(
            phi,
        )

        portmanteau = self._portmanteau(
            values,
            max_lag,
        )

        persistence = self._rolling_persistence(
            values,
            persistence_window,
        )

        return {
            "rho1": rho1,
            "phi": phi,
            "half_life": half_life,
            "portmanteau": portmanteau,
            "persistence": persistence,
        }
