import numpy as np
import pandas as pd

from statarb.signals.threshold import (
    empirical_frequency,
    optimize_gaussian_threshold,
    smooth_frequency,
)


def test_optimize_gaussian_threshold():

    result = optimize_gaussian_threshold()

    assert result.threshold > 0

    assert result.objective_value > 0


def test_empirical_frequency():

    zscore = pd.Series(
        [
            -3.0,
            -2.0,
            -1.0,
            0.0,
            1.0,
            2.0,
            3.0,
        ]
    )

    thresholds = np.array(
        [
            1.0,
            2.0,
        ]
    )

    frequency = empirical_frequency(
        zscore,
        thresholds,
    )

    # z > 1:
    # 2 values / 7 observations
    assert np.isclose(
        frequency[0],
        2 / 7,
    )

    # z > 2:
    # 1 value / 7 observations
    assert np.isclose(
        frequency[1],
        1 / 7,
    )


def test_smooth_frequency_shape():

    frequency = np.array(
        [
            0.5,
            0.3,
            0.2,
        ]
    )

    result = smooth_frequency(
        frequency,
        lam=1.0,
    )

    assert result.shape == frequency.shape

    assert np.isfinite(
        result,
    ).all()
