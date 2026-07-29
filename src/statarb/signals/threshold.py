from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm


@dataclass(frozen=True)
class ThresholdOptimizationResult:
    """
    Threshold optimization result.
    """

    threshold: float

    objective_value: float


def gaussian_profit_objective(
    threshold: float,
) -> float:
    """
    Gaussian parametric profit objective.

    Maximize:

        (1 - Phi(s0)) * s0

    Parameters
    ----------
    threshold:
        z-score threshold.

    Returns
    -------
    float
        Negative profit because optimizer minimizes.
    """

    frequency = 1 - norm.cdf(
        threshold,
    )

    profit = frequency * threshold

    return -float(profit)


def optimize_gaussian_threshold(
    bounds: tuple[float, float] = (0.0, 5.0),
) -> ThresholdOptimizationResult:
    """
    Optimize threshold under Gaussian assumption.

    s0* =
        argmax
        (1 - Phi(s0)) s0
    """

    result = minimize_scalar(
        gaussian_profit_objective,
        bounds=bounds,
        method="bounded",
    )

    return ThresholdOptimizationResult(
        threshold=float(result.x),
        objective_value=float(-result.fun),
    )


def empirical_frequency(
    zscore: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    Estimate empirical trading frequency.

    For each threshold:

        f_j =
        1/T sum 1{z_t > s0_j}

    One direction only.
    """

    zscore = np.asarray(
        zscore,
        dtype=float,
    )

    thresholds = np.asarray(
        thresholds,
        dtype=float,
    )

    T = len(zscore)

    frequencies = np.array(
        [np.sum(zscore > threshold) / T for threshold in thresholds],
        dtype=float,
    )

    return frequencies


def smooth_frequency(
    frequencies: np.ndarray,
    *,
    lam: float = 10.0,
) -> np.ndarray:
    """
    Smooth empirical trading frequency.

    Solve:

        min_f
            ||f - f_bar||^2
            + lambda ||Df||^2

    Closed form:

        f*
        =
        (I + lambda D'D)^(-1) f_bar
    """

    frequencies = np.asarray(
        frequencies,
        dtype=float,
    )

    J = len(frequencies)

    if J <= 1:
        return frequencies.copy()

    D = np.zeros(
        (
            J - 1,
            J,
        )
    )

    for i in range(J - 1):
        D[i, i] = 1
        D[i, i + 1] = -1

    I = np.eye(J)

    A = I + lam * D.T @ D

    smoothed = np.linalg.solve(
        A,
        frequencies,
    )

    return smoothed


def optimize_empirical_threshold(
    zscore: np.ndarray,
    thresholds: np.ndarray,
    *,
    lam: float = 10.0,
) -> ThresholdOptimizationResult:
    """
    Optimize threshold using nonparametric approach.

    Maximize:

        s0_j * f_j
    """

    thresholds = np.asarray(
        thresholds,
        dtype=float,
    )

    frequencies = empirical_frequency(
        zscore,
        thresholds,
    )

    frequencies = smooth_frequency(
        frequencies,
        lam=lam,
    )

    profits = thresholds * frequencies

    index = np.argmax(
        profits,
    )

    return ThresholdOptimizationResult(
        threshold=float(thresholds[index]),
        objective_value=float(profits[index]),
    )
