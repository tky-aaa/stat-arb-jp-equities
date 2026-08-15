import warnings

import numpy as np
import pandas as pd
from numpy.exceptions import ComplexWarning
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from newstatarb.config.contract import JohansenResult


class JohansenTester:
    def _select_var_lag(
        self,
        log_prices: pd.DataFrame,
        maxlags: int,
    ) -> int:
        """
        Select VAR lag using AIC.
        """

        n_obs = len(log_prices)
        n_vars = log_prices.shape[1]

        max_allowed = max(
            1,
            int((n_obs - 1) / (n_vars**2 + n_vars)),
        )

        effective_maxlags = min(
            maxlags,
            max_allowed,
        )

        model = VAR(log_prices.to_numpy())

        try:
            result = model.select_order(
                maxlags=effective_maxlags,
            )

            var_lag = result.selected_orders["aic"]

        except (np.linalg.LinAlgError, ValueError):
            var_lag = 1

        if var_lag is None:
            var_lag = 1

        return max(
            int(var_lag),
            1,
        )

    def _to_real_array(self, values) -> np.ndarray:
        """
        Convert numerical complex values
        caused by floating point errors into real arrays.
        """

        values = np.real_if_close(values, tol=1000)

        return np.asarray(values, dtype=float)

    def estimate_cointegration(
        self, log_prices: pd.DataFrame, det_order: int = 0, maxlags: int = 10
    ) -> JohansenResult:
        """
        Estimate Johansen cointegration relationship.
        """

        log_prices = log_prices.dropna().copy()

        if len(log_prices) < 30:
            raise ValueError("Too few observations for Johansen test.")

        tickers = list(log_prices.columns)

        var_lag = self._select_var_lag(log_prices, maxlags)

        k_ar_diff = max(var_lag - 1, 0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ComplexWarning)

            # Johansen test operates on numerical arrays only.
            # Preserve ticker/date metadata outside the estimation step.
            result = coint_johansen(
                log_prices.to_numpy(),
                det_order=det_order,
                k_ar_diff=k_ar_diff,
            )

        rank = 0

        for i in range(len(result.lr1)):
            if result.lr1[i] > result.cvt[i, 1]:
                rank += 1

            else:
                break

        if rank == 0:
            beta = np.empty((len(tickers), 0))

        else:
            beta = result.evec[:, :rank].copy()

            beta = self._to_real_array(beta)

            for i in range(rank):
                if beta[0, i] != 0:
                    beta[:, i] /= beta[0, i]

        return JohansenResult(
            rank=rank,
            tickers=tickers,
            beta=self._to_real_array(beta),
            eigenvalues=self._to_real_array(result.eig),
            trace_statistics=self._to_real_array(result.lr1),
            critical_values=np.asarray(result.cvt, dtype=float),
            var_lag=var_lag,
            k_ar_diff=k_ar_diff,
        )
