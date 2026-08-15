import pandas as pd

from statarb.cointegration.selection import (
    SelectedSpread,
    select_top_spreads,
)


def test_select_top_spreads():

    ranked_results = pd.DataFrame(
        {
            "tickers": [
                ("A", "B"),
                ("C", "D"),
                ("E", "F"),
            ],
            "beta": [
                [1.0, -1.0],
                [1.0, -0.5],
                [1.0, -2.0],
            ],
            "rank": [
                1,
                1,
                1,
            ],
            "beta_index": [
                0,
                0,
                0,
            ],
            "score": [
                1,
                2,
                3,
            ],
            "half_life": [
                5,
                10,
                20,
            ],
            "persistence": [
                0.9,
                0.8,
                0.7,
            ],
        }
    )

    selected = select_top_spreads(
        ranked_results,
        n_spreads=2,
    )

    assert len(selected) == 2

    assert isinstance(
        selected[0],
        SelectedSpread,
    )

    first = selected[0]

    assert first.tickers == [
        "A",
        "B",
    ]

    assert first.beta == [
        1.0,
        -1.0,
    ]

    assert first.rank == 1

    assert first.beta_index == 0

    assert first.score == 1.0

    assert first.half_life == 5.0

    assert first.persistence == 0.9


def test_select_top_spreads_empty():

    ranked_results = pd.DataFrame(
        columns=[
            "tickers",
            "beta",
            "rank",
            "beta_index",
            "score",
            "half_life",
            "persistence",
        ]
    )

    selected = select_top_spreads(
        ranked_results,
    )

    assert selected == []


def test_select_top_spreads_beta_conversion():

    ranked_results = pd.DataFrame(
        {
            "tickers": [
                ("A", "B"),
            ],
            "beta": [
                [
                    1,
                    -1,
                ],
            ],
            "rank": [
                1,
            ],
            "beta_index": [
                0,
            ],
            "score": [
                1,
            ],
            "half_life": [
                5,
            ],
            "persistence": [
                0.9,
            ],
        }
    )

    selected = select_top_spreads(
        ranked_results,
    )

    assert selected[0].beta == [
        1.0,
        -1.0,
    ]

    assert selected[0].rank == 1

    assert selected[0].beta_index == 0


def test_select_multi_vector_rank():
    """
    Test that multiple cointegration vectors
    can be selected independently.

    Example:

        rank=2

        beta_0
        beta_1

    are different spreads.
    """

    ranked_results = pd.DataFrame(
        {
            "tickers": [
                ("A", "B", "C"),
                ("A", "B", "C"),
            ],
            "beta": [
                [1.0, -1.0, 0.2],
                [1.0, 0.3, -1.2],
            ],
            "rank": [
                2,
                2,
            ],
            "beta_index": [
                0,
                1,
            ],
            "score": [
                1,
                2,
            ],
            "half_life": [
                5,
                10,
            ],
            "persistence": [
                0.9,
                0.8,
            ],
        }
    )

    selected = select_top_spreads(
        ranked_results,
        n_spreads=2,
    )

    assert len(selected) == 2

    assert selected[0].tickers == [
        "A",
        "B",
        "C",
    ]

    assert selected[0].beta_index == 0

    assert selected[1].beta_index == 1
