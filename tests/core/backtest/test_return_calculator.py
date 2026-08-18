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
        beta=pd.DataFrame(
            [
                [1.0, -1.0],
                [1.0, -1.0],
                [0.5, -1.0],
                [0.5, -1.0],
            ],
            index=index,
            columns=["AAA", "BBB"],
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

    signal = make_signal()

    beta = signal.beta
    normalized_beta = beta.div(
        beta.abs().sum(axis=1),
        axis=0,
    )

    spread_return = (
        pd.DataFrame(
            {
                "AAA": aaa_return,
                "BBB": bbb_return,
            },
        )
        .mul(
            normalized_beta,
        )
        .sum(axis=1)
    )

    expected = (signal.position.shift(1) * spread_return).fillna(0.0)

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


def test_calculate_uses_time_varying_beta() -> None:
    prices = make_prices()

    signal = make_signal()

    signal = Signal(
        spread=signal.spread,
        zscore=signal.zscore,
        position=signal.position,
        beta=pd.DataFrame(
            [
                [1.0, -1.0],
                [0.5, -1.0],
                [2.0, -1.0],
                [1.0, -1.0],
            ],
            index=prices.index,
            columns=["AAA", "BBB"],
        ),
    )

    result = ReturnCalculator().calculate(
        prices,
        make_analysis(),
        signal,
    )

    asset_returns = prices.pct_change()

    beta = signal.beta
    normalized_beta = beta.div(
        beta.abs().sum(axis=1),
        axis=0,
    )

    spread_return = asset_returns.mul(
        normalized_beta,
    ).sum(axis=1)

    expected = (signal.position.shift(1) * spread_return).fillna(0.0)

    expected.name = "pnl"

    pd.testing.assert_series_equal(
        result,
        expected,
    )
