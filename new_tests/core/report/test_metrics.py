import pandas as pd
import pytest

from newstatarb.config.contract import (
    BacktestResult,
    Signal,
)
from newstatarb.core.report.metrics import (
    BacktestMetricsCalculator,
)


def make_result() -> BacktestResult:
    index = pd.date_range(
        "2025-01-01",
        periods=4,
    )

    pnl = pd.Series(
        [0.0, 1.0, -0.5, 1.0],
        index=index,
        name="portfolio_return",
    )

    cumulative_pnl = pnl.cumsum()

    equity = 100.0 + cumulative_pnl

    return BacktestResult(
        pnl=pnl,
        cumulative_pnl=cumulative_pnl,
        equity=equity,
    )


def make_signal(
    positions: list[float],
) -> Signal:
    index = pd.date_range(
        "2025-01-01",
        periods=len(positions),
    )

    return Signal(
        spread=pd.Series(
            [10.0] * len(positions),
            index=index,
        ),
        zscore=pd.Series(
            [0.0] * len(positions),
            index=index,
        ),
        position=pd.Series(
            positions,
            index=index,
        ),
    )


def test_calculate_metrics() -> None:
    result = make_result()

    signals = [
        make_signal([0.0, 1.0, 1.0, 0.0]),
        make_signal([0.0, -1.0, 0.0, 0.0]),
    ]

    metrics = BacktestMetricsCalculator().calculate(
        result,
        signals,
    )

    assert metrics.total_return == pytest.approx(1.5 / 100)
    assert metrics.volatility > 0
    assert metrics.sharpe_ratio != 0
    assert metrics.max_drawdown < 0
    assert metrics.win_rate == pytest.approx(2 / 3)
    assert metrics.number_of_trades == 4


def test_calculate_rejects_nonpositive_periods_per_year() -> None:
    with pytest.raises(
        ValueError,
        match="periods_per_year must be positive",
    ):
        BacktestMetricsCalculator().calculate(
            make_result(),
            [],
            periods_per_year=0,
        )


def test_calculate_rejects_empty_result() -> None:
    index = pd.DatetimeIndex([])

    result = BacktestResult(
        pnl=pd.Series(
            dtype=float,
            index=index,
        ),
        cumulative_pnl=pd.Series(
            dtype=float,
            index=index,
        ),
        equity=pd.Series(
            dtype=float,
            index=index,
        ),
    )

    with pytest.raises(
        ValueError,
        match="Backtest result must not be empty",
    ):
        BacktestMetricsCalculator().calculate(
            result,
            [],
        )


def test_calculate_rejects_zero_initial_equity() -> None:
    index = pd.date_range(
        "2025-01-01",
        periods=2,
    )

    pnl = pd.Series(
        [0.0, 1.0],
        index=index,
    )

    result = BacktestResult(
        pnl=pnl,
        cumulative_pnl=pnl.cumsum(),
        equity=pd.Series(
            [0.0, 1.0],
            index=index,
        ),
    )

    with pytest.raises(
        ValueError,
        match="Initial equity must not be zero",
    ):
        BacktestMetricsCalculator().calculate(
            result,
            [],
        )
