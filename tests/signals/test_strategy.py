import pandas as pd

from statarb.signals.strategy import (
    generate_linear_signal,
    generate_threshold_signal,
)

# ======================================================
# Test linear strategy
# ======================================================


def test_generate_linear_signal():

    zscore = pd.Series(
        [
            -2.0,
            -1.0,
            0.0,
            1.0,
            2.0,
            5.0,
        ]
    )

    signal = generate_linear_signal(
        zscore,
        threshold=2.0,
    )

    expected = pd.Series(
        [
            1.0,
            0.5,
            0.0,
            -0.5,
            -1.0,
            -1.0,
        ]
    )

    pd.testing.assert_series_equal(
        signal,
        expected,
    )


# ======================================================
# Test threshold strategy
# ======================================================


def test_generate_threshold_signal():

    zscore = pd.Series(
        [
            0.0,
            -2.5,
            -1.5,
            0.2,
            2.5,
            1.0,
            -2.5,
            0.0,
        ]
    )

    signal = generate_threshold_signal(
        zscore,
        entry_threshold=2.0,
        exit_threshold=0.5,
    )

    expected = pd.Series(
        [
            0,
            1,
            1,
            0,
            -1,
            -1,
            0,
            0,
        ],
        name="signal",
    )

    pd.testing.assert_series_equal(
        signal,
        expected,
    )


# ======================================================
# Test invalid threshold
# ======================================================


def test_invalid_threshold():

    zscore = pd.Series(
        [
            0.0,
            1.0,
        ]
    )

    try:
        generate_linear_signal(
            zscore,
            threshold=0,
        )

        assert False

    except ValueError:
        pass

    try:
        generate_threshold_signal(
            zscore,
            entry_threshold=-1,
            exit_threshold=0.5,
        )

        assert False

    except ValueError:
        pass

    try:
        generate_threshold_signal(
            zscore,
            entry_threshold=1.0,
            exit_threshold=1.0,
        )

        assert False

    except ValueError:
        pass
