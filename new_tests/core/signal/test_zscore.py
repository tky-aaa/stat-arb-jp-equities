import numpy as np
import pandas as pd
import pytest

from newstatarb.core.signal.zscore import ZScoreCalculator


@pytest.fixture
def spread() -> pd.Series:
    index = pd.date_range(
        "2025-01-01",
        periods=5,
        freq="D",
    )

    return pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        index=index,
        name="spread",
    )


def test_calculate_returns_rolling_zscore(
    spread: pd.Series,
) -> None:
    calculator = ZScoreCalculator()

    zscore = calculator.calculate(
        spread,
        window=3,
    )

    expected = pd.Series(
        [
            np.nan,
            np.nan,
            1.0,
            1.0,
            1.0,
        ],
        index=spread.index,
        name="zscore",
    )

    pd.testing.assert_series_equal(
        zscore,
        expected,
    )


def test_calculate_preserves_index_and_name(
    spread: pd.Series,
) -> None:
    calculator = ZScoreCalculator()

    zscore = calculator.calculate(
        spread,
        window=3,
    )

    pd.testing.assert_index_equal(
        zscore.index,
        spread.index,
    )

    assert zscore.name == "zscore"


def test_calculate_rejects_invalid_window(
    spread: pd.Series,
) -> None:
    calculator = ZScoreCalculator()

    with pytest.raises(
        ValueError,
        match="window must be greater than 1",
    ):
        calculator.calculate(
            spread,
            window=1,
        )

    with pytest.raises(
        ValueError,
        match="window must be greater than 1",
    ):
        calculator.calculate(
            spread,
            window=0,
        )

    with pytest.raises(
        ValueError,
        match="window must be greater than 1",
    ):
        calculator.calculate(
            spread,
            window=-1,
        )
