import pandas as pd

from statarb.config.contract import Signal
from statarb.core.backtest.return_calculator import ReturnCalculator


def make_signal() -> Signal:
    index = pd.date_range(
        "2025-01-01",
        periods=4,
    )

    return Signal(
        spread=pd.Series(
            [10.0, 11.0, 9.0, 10.0],
            index=index,
        ),
        zscore=pd.Series(
            [0.0, 1.0, -1.0, 0.5],
            index=index,
        ),
        position=pd.Series(
            [0.0, 1.0, -1.0, 1.0],
            index=index,
        ),
    )


def test_calculate_uses_lagged_position() -> None:
    result = ReturnCalculator().calculate(
        make_signal(),
    )

    expected = pd.Series(
        [
            None,
            0.0,
            -2.0,
            -1.0,
        ],
        index=pd.date_range(
            "2025-01-01",
            periods=4,
        ),
        name="pnl",
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_calculate_preserves_signal_index() -> None:
    signal = make_signal()

    result = ReturnCalculator().calculate(signal)

    assert result.index.equals(signal.spread.index)


def test_calculate_preserves_pnl_name() -> None:
    result = ReturnCalculator().calculate(
        make_signal(),
    )

    assert result.name == "pnl"
