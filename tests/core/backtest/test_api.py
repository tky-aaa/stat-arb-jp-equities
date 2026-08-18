import pandas as pd
import pytest

from statarb.config.contract import (
    CointegrationAnalysis,
    PortfolioDecision,
    Prices,
    Signal,
    Spread,
    SpreadEvaluation,
)
from statarb.core.backtest.api import BacktestAPI


def make_analysis() -> CointegrationAnalysis:
    index = pd.date_range(
        "2025-01-01",
        periods=4,
    )

    spread = Spread(
        tickers=["AAA", "BBB"],
        beta=pd.DataFrame(
            [[1.0, -1.0]] * len(index),
            index=index,
            columns=["AAA", "BBB"],
        ),
        intercept=pd.Series(
            0.0,
            index=index,
            name="intercept",
        ),
        values=pd.Series(
            [10.0, 11.0, 9.0, 10.0],
            index=index,
        ),
    )

    evaluation = SpreadEvaluation(
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
    )

    return CointegrationAnalysis(
        rank=1,
        beta_index=0,
        spread=spread,
        evaluation=evaluation,
    )


def make_signal(
    spread_values: list[float],
    position_values: list[float],
) -> Signal:
    index = pd.date_range(
        "2025-01-01",
        periods=len(spread_values),
    )

    return Signal(
        spread=pd.Series(
            spread_values,
            index=index,
        ),
        zscore=pd.Series(
            [0.0] * len(spread_values),
            index=index,
        ),
        position=pd.Series(
            position_values,
            index=index,
        ),
    )


def make_prices(signal: Signal) -> Prices:
    return Prices(
        training=pd.DataFrame(
            {
                "AAA": [100.0] * len(signal.spread),
                "BBB": [100.0] * len(signal.spread),
            },
            index=signal.spread.index,
        ),
        test=pd.DataFrame(
            {
                "AAA": [100.0, 102.0, 100.0, 101.0][: len(signal.spread)],
                "BBB": [100.0] * len(signal.spread),
            },
            index=signal.spread.index,
        ),
    )


def test_service_calculates_portfolio_result() -> None:
    analysis = make_analysis()

    signal = make_signal(
        [10.0, 11.0, 9.0, 10.0],
        [0.0, 1.0, -1.0, 1.0],
    )

    prices = make_prices(signal)

    decision = PortfolioDecision(
        analyses=[analysis],
        signals=[signal],
        weights=[0.5],
    )

    result = BacktestAPI(
        initial_capital=100.0,
    ).service(
        prices,
        decision,
    )

    expected_pnl = pd.Series(
        [
            0.0,
            0.0,
            -0.004901960784313736,
            -0.0025,
        ],
        index=signal.spread.index,
        name="portfolio_return",
    )

    expected_cumulative_pnl = expected_pnl.cumsum()

    expected_equity = 100.0 + expected_cumulative_pnl

    pd.testing.assert_series_equal(
        result.pnl,
        expected_pnl,
    )

    pd.testing.assert_series_equal(
        result.cumulative_pnl,
        expected_cumulative_pnl,
    )

    pd.testing.assert_series_equal(
        result.equity,
        expected_equity,
    )


def test_service_rejects_mismatched_analyses_and_weights() -> None:
    analysis = make_analysis()

    signal = make_signal(
        [10.0, 11.0],
        [0.0, 1.0],
    )

    decision = PortfolioDecision(
        analyses=[analysis],
        signals=[signal],
        weights=[],
    )

    prices = make_prices(signal)

    with pytest.raises(
        ValueError,
        match="Number of analyses and weights must match",
    ):
        BacktestAPI().service(
            prices,
            decision,
        )


def test_service_rejects_mismatched_analyses_and_signals() -> None:
    analysis = make_analysis()
    signal = make_signal(
        [10.0, 11.0],
        [0.0, 1.0],
    )

    prices = make_prices(signal)
    decision = PortfolioDecision(
        analyses=[analysis],
        signals=[],
        weights=[1.0],
    )

    with pytest.raises(
        ValueError,
        match="Number of analyses and signals must match",
    ):
        BacktestAPI().service(
            prices,
            decision,
        )


def test_service_rejects_nonpositive_initial_capital() -> None:
    analysis = make_analysis()

    signal = make_signal(
        [10.0, 11.0],
        [0.0, 1.0],
    )

    decision = PortfolioDecision(
        analyses=[analysis],
        signals=[signal],
        weights=[1.0],
    )

    prices = make_prices(signal)

    with pytest.raises(
        ValueError,
        match="initial_capital must be positive",
    ):
        BacktestAPI(
            initial_capital=0.0,
        ).service(
            prices,
            decision,
        )
