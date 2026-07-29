import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import (
    adfuller,
    kpss,
)


@dataclass(frozen=True)
class SpreadEvaluation:
    """
    Statistics describing one spread.
    """

    # Stationarity

    adf_stat: float
    adf_pvalue: float

    kpss_stat: float
    kpss_pvalue: float

    # Mean reversion

    rho1: float
    phi: float
    half_life: float

    # Spread distribution

    mean: float
    variance: float
    std: float

    # Higher autocorrelation

    portmanteau: float


def _lag_autocorrelation(
    spread: pd.Series,
    lag: int,
) -> float:
    """
    Estimate lag autocorrelation.

    rho_k =
        Cov(z_t,z_{t-k})
        ----------------
        Var(z_t)
    """

    x = spread.dropna()

    x = x - x.mean()

    numerator = (x.iloc[:-lag] * x.iloc[lag:]).mean()

    denominator = (x**2).mean()

    if denominator == 0:
        return np.nan

    return float(numerator / denominator)


def _estimate_ar1_phi(
    spread: pd.Series,
) -> float:
    """
    Estimate AR(1) coefficient.

    z_t = phi z_{t-1} + eps_t
    """

    x = spread.dropna()

    lag = x.shift(1)

    df = pd.concat(
        [
            x,
            lag,
        ],
        axis=1,
    ).dropna()

    phi = np.polyfit(
        df.iloc[:, 1],
        df.iloc[:, 0],
        1,
    )[0]

    return float(phi)


def _estimate_half_life(
    phi: float,
) -> float:
    """
    Half-life implied by AR(1).

    HL =
        -log(2)
        --------
        log(phi)

    Only meaningful for
    0 < phi < 1.
    """

    if phi <= 0 or phi >= 1:
        return np.inf

    return float(-np.log(2) / np.log(phi))


def _portmanteau(
    spread: pd.Series,
    max_lag: int,
) -> float:
    """
    Portmanteau statistic.

    Sum of squared autocorrelations.
    """

    value = 0.0

    for lag in range(
        1,
        max_lag + 1,
    ):
        rho = _lag_autocorrelation(
            spread,
            lag,
        )

        value += rho**2

    return float(value)


def evaluate_spread(
    spread: pd.Series,
    *,
    max_lag: int = 10,
) -> SpreadEvaluation:
    """
    Evaluate one spread.

    Parameters
    ----------
    spread:
        Spread time series.

    max_lag:
        Maximum lag for portmanteau.


    Returns
    -------
    SpreadEvaluation
    """

    spread = spread.dropna().astype(float)

    if len(spread) < 50:
        raise ValueError("Too few observations.")

    # =========================
    # Stationarity tests
    # =========================

    adf_result = adfuller(
        spread,
        autolag="AIC",
    )

    adf_stat = adf_result[0]
    adf_pvalue = adf_result[1]

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            InterpolationWarning,
        )

        kpss_result = kpss(
            spread,
            regression="c",
            nlags="auto",
        )

    kpss_stat = kpss_result[0]
    kpss_pvalue = kpss_result[1]

    # =========================
    # Mean reversion
    # =========================

    rho1 = _lag_autocorrelation(
        spread,
        1,
    )

    phi = _estimate_ar1_phi(
        spread,
    )

    half_life = _estimate_half_life(
        phi,
    )

    # =========================
    # Distribution
    # =========================

    mean = float(spread.mean())

    variance = float(spread.var())

    std = float(spread.std())

    # =========================
    # Higher lag autocorrelation
    # =========================

    port = _portmanteau(
        spread,
        max_lag,
    )

    return SpreadEvaluation(
        adf_stat=adf_stat,
        adf_pvalue=adf_pvalue,
        kpss_stat=kpss_stat,
        kpss_pvalue=kpss_pvalue,
        rho1=rho1,
        phi=phi,
        half_life=half_life,
        mean=mean,
        variance=variance,
        std=std,
        portmanteau=port,
    )
