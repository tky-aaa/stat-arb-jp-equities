import pandas as pd

from statarb.config.contract import (
    CointegrationAnalysis,
    Signal,
    Spread,
    SpreadEvaluation,
)
from statarb.core.backtest.return_calculator import ReturnCalculator


def make_analysis() -> CointegrationAnalysis:
    return CointegrationAnalysis(
        rank=1,
        beta_index=0,
        spread=Spread(
            tickers=["AAA", "BBB"],
            beta=pd.DataFrame(
                [[1.0, -1.0]],
                columns=["AAA", "BBB"],
            ),
            intercept=pd.Series(
                [0.0],
                name="intercept",
            ),
            values=pd.Series(dtype=float),
        ),
        evaluation=SpreadEvaluation(
            adf_stat=0.0,
            adf_pvalue=0.01,
            kpss_stat=0.0,
            kpss_pvalue=0.10,
            rho1=0.0,
            phi=0.0,
            half_life=1.0,
            persistence=0.0,
            mean=0.0,
            variance=1.0,
            std=1.0,
            portmanteau=1.0,
        ),
    )


def make_prices() -> pd.DataFrame:
    index = pd.date_range(
        "2025-01-01",
        periods=4,
    )

    return pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 99.0, 100.0],
            "BBB": [100.0, 99.0, 101.0, 100.0],
        },
        index=index,
    )


def make_signal() -> Signal:
    index = pd.date_range(
        "2025-01-01",
        periods=4,
    )

    return Signal(
        spread=pd.Series(
            [0.0, 1.0, -1.0, 0.0],
            index=index,
        ),
        zscore=pd.Series(
            [0.0, 1.0, -1.0, 0.0],
            index=index,
        ),
        position=pd.Series(
            [0.0, 1.0, -1.0, 1.0],
            index=index,
        ),
    )


def test_calculate_uses_normalized_beta_and_lagged_position() -> None:
    result = ReturnCalculator().calculate(
        make_prices(),
        make_analysis(),
        make_signal(),
    )

    aaa_return = make_prices()["AAA"].pct_change()
    bbb_return = make_prices()["BBB"].pct_change()

    spread_return = 0.5 * aaa_return - 0.5 * bbb_return

    expected = (make_signal().position.shift(1) * spread_return).fillna(0.0)

    expected.name = "pnl"

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_calculate_preserves_signal_index() -> None:
    signal = make_signal()

    result = ReturnCalculator().calculate(
        make_prices(),
        make_analysis(),
        signal,
    )

    assert result.index.equals(signal.position.index)


def test_calculate_preserves_pnl_name() -> None:
    result = ReturnCalculator().calculate(
        make_prices(),
        make_analysis(),
        make_signal(),
    )

    assert result.name == "pnl"
