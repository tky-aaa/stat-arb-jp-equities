from dataclasses import dataclass

import numpy as np
import pandas as pd

from .threshold import (
    optimize_empirical_threshold,
    optimize_gaussian_threshold,
)


@dataclass(frozen=True)
class SignalThreshold:
    """
    Selected signal threshold.
    """

    threshold: float

    method: str

    objective_value: float


def select_gaussian_threshold() -> SignalThreshold:
    """
    Select threshold using Gaussian assumption.
    """

    result = optimize_gaussian_threshold()

    return SignalThreshold(
        threshold=result.threshold,
        method="gaussian",
        objective_value=result.objective_value,
    )


def select_empirical_threshold(
    zscore: pd.Series,
    *,
    n_grid: int = 100,
    min_threshold: float = 0.1,
    max_threshold: float = 5.0,
    lam: float = 10.0,
) -> SignalThreshold:
    """
    Select threshold using empirical frequency.
    """

    thresholds = np.linspace(
        min_threshold,
        max_threshold,
        n_grid,
    )

    result = optimize_empirical_threshold(
        zscore.values,
        thresholds,
        lam=lam,
    )

    return SignalThreshold(
        threshold=result.threshold,
        method="empirical",
        objective_value=result.objective_value,
    )
