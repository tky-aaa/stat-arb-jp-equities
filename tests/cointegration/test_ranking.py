import pandas as pd

from statarb.cointegration.ranking import (
    rank_spreads,
)


def test_rank_spreads_filters_and_ranks():

    evaluation_results = pd.DataFrame(
        {
            "tickers": [
                ("A", "B"),
                ("C", "D"),
                ("E", "F"),
                ("G", "H"),
            ],
            "beta": [
                [1.0, -1.0],
                [1.0, -1.0],
                [1.0, -1.0],
                [1.0, -1.0],
            ],
            "adf_pvalue": [
                0.01,
                0.01,
                0.20,
                0.01,
            ],
            "kpss_pvalue": [
                0.10,
                0.20,
                0.10,
                0.01,
            ],
            "half_life": [
                5.0,
                10.0,
                3.0,
                2.0,
            ],
            "persistence": [
                0.8,
                0.9,
                1.0,
                0.95,
            ],
        }
    )

    result = rank_spreads(
        evaluation_results,
    )

    # =========================
    # Filtering
    # =========================

    # Keep only:
    #
    # A,B
    # C,D
    #
    # Remove:
    #
    # E,F -> ADF fail
    # G,H -> KPSS fail

    assert len(result) == 2

    assert ("A", "B") in result["tickers"].tolist()

    assert ("C", "D") in result["tickers"].tolist()

    # =========================
    # Ranking
    # =========================

    # A,B:
    # half_life rank = 1
    # persistence rank = 2
    # score = 3
    #
    # C,D:
    # half_life rank = 2
    # persistence rank = 1
    # score = 3
    #
    # tie -> shorter half-life first

    assert tuple(result.iloc[0]["tickers"]) == ("A", "B")

    assert result.iloc[0]["half_life"] == 5.0

    assert result.iloc[0]["score"] == 3


def test_rank_spreads_empty_after_filter():

    evaluation_results = pd.DataFrame(
        {
            "adf_pvalue": [
                0.5,
            ],
            "kpss_pvalue": [
                0.01,
            ],
            "half_life": [
                5.0,
            ],
            "persistence": [
                0.8,
            ],
        }
    )

    result = rank_spreads(
        evaluation_results,
    )

    assert len(result) == 0
