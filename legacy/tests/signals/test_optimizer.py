import pandas as pd

from statarb.signals.optimizer import (
    select_empirical_threshold,
    select_gaussian_threshold,
)


def test_select_gaussian_threshold():

    result = select_gaussian_threshold()

    assert result.threshold > 0

    assert result.method == "gaussian"

    assert result.objective_value > 0


def test_select_empirical_threshold():

    zscore = pd.Series(
        [
            -3.0,
            -2.5,
            -1.0,
            0.0,
            1.0,
            2.5,
            3.0,
        ]
    )

    result = select_empirical_threshold(
        zscore,
    )

    assert result.threshold > 0

    assert result.method == "empirical"

    assert result.objective_value > 0
