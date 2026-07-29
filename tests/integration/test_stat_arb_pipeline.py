import pandas as pd

from statarb.cointegration.selection import (
    select_top_spreads,
)


def test_selected_spread_to_backtest_pipeline():

    johansen_results = pd.DataFrame(
        {
            "tickers": [
                [
                    "A",
                    "B",
                ],
                [
                    "C",
                    "D",
                ],
            ],
            "rank": [
                1,
                1,
            ],
            "beta": [
                [
                    1.0,
                    -1.0,
                ],
                [
                    1.0,
                    -1.0,
                ],
            ],
            "beta_index": [
                0,
                0,
            ],
            "score": [
                1.0,
                2.0,
            ],
            "half_life": [
                5.0,
                10.0,
            ],
            "persistence": [
                0.9,
                0.8,
            ],
        }
    )

    selected = select_top_spreads(
        johansen_results,
        n_spreads=1,
    )

    assert len(selected) == 1

    assert selected[0].tickers == [
        "A",
        "B",
    ]

    assert selected[0].rank == 1

    assert selected[0].beta_index == 0

    assert selected[0].beta == [
        1.0,
        -1.0,
    ]
