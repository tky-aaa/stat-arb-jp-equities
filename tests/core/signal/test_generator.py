import numpy as np
import pandas as pd
import pytest

from statarb.core.signal.generator import SignalGenerator


@pytest.fixture
def zscore() -> pd.Series:
    index = pd.date_range(
        "2025-01-01",
        periods=7,
        freq="D",
    )

    return pd.Series(
        [
            0.0,
            2.5,
            1.0,
            0.4,
            -2.5,
            -0.4,
            0.0,
        ],
        index=index,
        name="zscore",
    )


def test_generate_enters_and_exits_both_positions(
    zscore: pd.Series,
) -> None:
    generator = SignalGenerator()

    position = generator.generate(
        zscore,
        available=pd.Series(True, index=zscore.index),
        entry_threshold=2.0,
        exit_threshold=0.5,
    )

    expected = pd.Series(
        [
            0,
            -1,
            -1,
            0,
            1,
            0,
            0,
        ],
        index=zscore.index,
        name="position",
    )

    pd.testing.assert_series_equal(
        position,
        expected,
    )


def test_generate_enters_and_exits_short_position() -> None:
    zscore = pd.Series(
        [
            0.0,
            2.5,
            2.0,
            0.5,
            0.4,
        ],
        index=pd.date_range(
            "2025-01-01",
            periods=5,
            freq="D",
        ),
        name="zscore",
    )

    generator = SignalGenerator()

    position = generator.generate(
        zscore,
        available=pd.Series(True, index=zscore.index),
        entry_threshold=2.0,
        exit_threshold=0.5,
    )

    expected = pd.Series(
        [
            0,
            -1,
            -1,
            0,
            0,
        ],
        index=zscore.index,
        name="position",
    )

    pd.testing.assert_series_equal(
        position,
        expected,
    )


def test_generate_preserves_position_during_nan() -> None:
    zscore = pd.Series(
        [
            2.5,
            np.nan,
            1.0,
            0.4,
        ],
        index=pd.date_range(
            "2025-01-01",
            periods=4,
            freq="D",
        ),
        name="zscore",
    )

    generator = SignalGenerator()

    position = generator.generate(
        zscore,
        available=pd.Series(True, index=zscore.index),
        entry_threshold=2.0,
        exit_threshold=0.5,
    )

    expected = pd.Series(
        [
            -1,
            -1,
            -1,
            0,
        ],
        index=zscore.index,
        name="position",
    )

    pd.testing.assert_series_equal(
        position,
        expected,
    )


def test_generate_preserves_index_and_name(
    zscore: pd.Series,
) -> None:
    generator = SignalGenerator()

    position = generator.generate(
        zscore,
        available=pd.Series(True, index=zscore.index),
        entry_threshold=2.0,
        exit_threshold=0.5,
    )

    pd.testing.assert_index_equal(
        position.index,
        zscore.index,
    )

    assert position.name == "position"


def test_generate_rejects_invalid_thresholds(
    zscore: pd.Series,
) -> None:
    generator = SignalGenerator()

    with pytest.raises(
        ValueError,
        match="entry_threshold must be positive",
    ):
        generator.generate(
            zscore,
            available=pd.Series(True, index=zscore.index),
            entry_threshold=0.0,
            exit_threshold=0.5,
        )

    with pytest.raises(
        ValueError,
        match="exit_threshold must be non-negative",
    ):
        generator.generate(
            zscore,
            available=pd.Series(True, index=zscore.index),
            entry_threshold=2.0,
            exit_threshold=-0.1,
        )

    with pytest.raises(
        ValueError,
        match="exit_threshold must be smaller than entry_threshold",
    ):
        generator.generate(
            zscore,
            available=pd.Series(True, index=zscore.index),
            entry_threshold=2.0,
            exit_threshold=2.0,
        )


def test_generate_closes_position_when_unavailable() -> None:
    index = pd.date_range(
        "2025-01-01",
        periods=4,
        freq="D",
    )
    zscore = pd.Series(
        [2.5, 1.0, 1.0, 2.5],
        index=index,
        name="zscore",
    )
    available = pd.Series(
        [True, False, True, True],
        index=index,
    )

    generator = SignalGenerator()
    position = generator.generate(
        zscore,
        available=available,
        entry_threshold=2.0,
        exit_threshold=0.5,
    )

    expected = pd.Series(
        [-1, 0, 0, -1],
        index=index,
        name="position",
    )

    pd.testing.assert_series_equal(
        position,
        expected,
    )
