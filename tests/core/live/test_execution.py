import pandas as pd

from statarb.config.contract import (
    CointegrationAnalysis,
    PortfolioDecision,
    Signal,
)
from statarb.core.live.execution import LiveExecutionBuilder


def make_decision(position: int) -> PortfolioDecision:
    analysis = CointegrationAnalysis(
        tickers=["A", "B"],
        rank=1,
        beta_index=0,
        beta=pd.Series([1.0, -0.5]),
        spread=None,
        evaluation=None,
    )

    signal = Signal(
        spread=pd.Series([0.0]),
        zscore=pd.Series([0.0]),
        position=pd.Series([position]),
    )

    return PortfolioDecision(
        analyses=[analysis],
        signals=[signal],
        weights=[1.0],
    )


def test_builds_long_orders() -> None:
    decision = make_decision(1)

    execution = LiveExecutionBuilder(
        base_shares=100,
    ).build(decision)

    assert len(execution.orders) == 2

    assert execution.orders[0].ticker == "A"
    assert execution.orders[0].quantity == 100
    assert execution.orders[0].side == "BUY"

    assert execution.orders[1].ticker == "B"
    assert execution.orders[1].quantity == 50
    assert execution.orders[1].side == "SELL"


def test_builds_short_orders() -> None:
    decision = make_decision(-1)

    execution = LiveExecutionBuilder(
        base_shares=100,
    ).build(decision)

    assert len(execution.orders) == 2

    assert execution.orders[0].side == "SELL"
    assert execution.orders[1].side == "BUY"


def test_does_not_create_orders_when_position_is_zero() -> None:
    decision = make_decision(0)

    execution = LiveExecutionBuilder().build(decision)

    assert execution.orders == []
