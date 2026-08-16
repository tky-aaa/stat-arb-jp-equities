import numpy as np
import pandas as pd
import pytest

from statarb.core.signal.threshold.empirical import (
    EmpiricalThresholdOptimizer,
)


@pytest.fixture
def zscore() -> pd.Series:
    return pd.Series(
        [
            -2.5,
            -1.5,
            -0.5,
            0.5,
            1.5,
            2.5,
        ],
        dtype=float,
    )


def test_optimize_returns_threshold_from_grid(
    zscore: pd.Series,
) -> None:
    optimizer = EmpiricalThresholdOptimizer(
        min_threshold=0.5,
        max_threshold=2.5,
        n_thresholds=5,
        smoothing_lambda=0.0,
    )

    threshold = optimizer.optimize(zscore)

    assert threshold in np.linspace(
        0.5,
        2.5,
        5,
    )


def test_rejects_empty_zscore() -> None:
    optimizer = EmpiricalThresholdOptimizer(
        min_threshold=0.5,
        max_threshold=2.5,
        n_thresholds=5,
        smoothing_lambda=0.0,
    )

    with pytest.raises(
        ValueError,
        match="No valid z-score observations",
    ):
        optimizer.optimize(
            pd.Series(
                [np.nan, np.nan],
            )
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "min_threshold": 0.0,
            "max_threshold": 2.0,
            "n_thresholds": 5,
            "smoothing_lambda": 1.0,
        },
        {
            "min_threshold": 2.0,
            "max_threshold": 2.0,
            "n_thresholds": 5,
            "smoothing_lambda": 1.0,
        },
        {
            "min_threshold": 0.5,
            "max_threshold": 2.0,
            "n_thresholds": 1,
            "smoothing_lambda": 1.0,
        },
        {
            "min_threshold": 0.5,
            "max_threshold": 2.0,
            "n_thresholds": 5,
            "smoothing_lambda": -1.0,
        },
    ],
)
def test_rejects_invalid_parameters(
    kwargs: dict,
) -> None:
    with pytest.raises(ValueError):
        EmpiricalThresholdOptimizer(
            **kwargs,
        )
