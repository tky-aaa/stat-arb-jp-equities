import pandas as pd

from statarb.execution.paper import simulate_paper_execution


def test_simulate_paper_execution():

    index = pd.date_range(
        "2024-01-01",
        periods=6,
    )

    spread = pd.Series(
        [0.1, 0.3, -0.2, -0.4, 0.1, 0.2],
        index=index,
    )

    signal = pd.Series(
        [0, 1, 1, -1, -1, 0],
        index=index,
    )

    trades = simulate_paper_execution(
        spread,
        signal,
    )

    assert len(trades) == 3

    assert list(trades["action"]) == [
        "BUY_SPREAD",
        "SELL_SPREAD",
        "EXIT_SHORT",
    ]

    assert list(trades["position"]) == [
        1,
        -1,
        0,
    ]
