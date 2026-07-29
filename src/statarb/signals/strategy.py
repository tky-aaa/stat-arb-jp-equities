from __future__ import annotations

import numpy as np
import pandas as pd

from .optimizer import (
    select_empirical_threshold,
    select_gaussian_threshold,
)


def generate_linear_signal(
    zscore: pd.Series,
    *,
    threshold: float = 1.0,
) -> pd.Series:
    """
    Linear mean-reversion sizing signal.

    Palomar:

        s_t = -zscore_t / s0

    The signal is clipped to [-1, 1].
    """

    if threshold <= 0:
        raise ValueError("threshold must be positive.")

    signal = -zscore / threshold

    return signal.clip(
        lower=-1,
        upper=1,
    )


def generate_threshold_signal(
    zscore: pd.Series,
    *,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
) -> pd.Series:
    """
    Threshold mean-reversion strategy.

    Entry
    -----

    zscore > entry_threshold:
        short spread (-1)

    zscore < -entry_threshold:
        long spread (+1)


    Exit
    ----

    long position:
        exit when zscore >= -exit_threshold

    short position:
        exit when zscore <= exit_threshold
    """

    if entry_threshold <= 0:
        raise ValueError("entry_threshold must be positive.")

    if exit_threshold < 0:
        raise ValueError("exit_threshold must be non-negative.")

    if exit_threshold >= entry_threshold:
        raise ValueError("exit_threshold must be smaller than entry_threshold.")

    position = 0

    signals = []

    for value in zscore:
        if np.isnan(value):
            signals.append(position)
            continue

        # Entry

        if position == 0:
            if value > entry_threshold:
                position = -1

            elif value < -entry_threshold:
                position = 1

        # Exit

        elif (
            position == 1
            and value >= -exit_threshold
            or position == -1
            and value <= exit_threshold
        ):
            position = 0

        signals.append(position)

    return pd.Series(
        signals,
        index=zscore.index,
        name="signal",
    )


def generate_optimized_threshold_signal(
    zscore: pd.Series,
    *,
    method: str = "empirical",
    lam: float = 10.0,
    exit_ratio: float = 0.25,
) -> pd.Series:
    """
    Generate threshold signal using optimized entry threshold.

    Parameters
    ----------
    zscore:
        Rolling z-score series.

    method:
        Threshold optimization method.

        gaussian:
            Parametric Gaussian approach.

        empirical:
            Data-driven approach.

    lam:
        Smoothness parameter for empirical optimization.

    exit_ratio:
        Exit threshold ratio relative to entry threshold.
    """

    if exit_ratio < 0 or exit_ratio >= 1:
        raise ValueError("exit_ratio must be in [0,1).")

    if method == "gaussian":
        result = select_gaussian_threshold()

    elif method == "empirical":
        result = select_empirical_threshold(
            zscore,
            lam=lam,
        )

    else:
        raise ValueError("method must be 'gaussian' or 'empirical'.")

    entry_threshold = result.threshold

    exit_threshold = exit_ratio * entry_threshold

    return generate_threshold_signal(
        zscore,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
    )
