import numpy as np
import pandas as pd
import pytest

from statarb.config.contract import (
    CointegrationAnalysis,
    Spread,
    SpreadEvaluation,
)
from statarb.core.portfolio.selector import PortfolioSelector


def make_analysis(
    *,
    half_life: float,
    portmanteau: float,
    adf_pvalue: float = 0.01,
    kpss_pvalue: float = 0.10,
) -> CointegrationAnalysis:
    evaluation = SpreadEvaluation(
        adf_stat=0.0,
        adf_pvalue=adf_pvalue,
        kpss_stat=0.0,
        kpss_pvalue=kpss_pvalue,
        rho1=0.0,
        phi=0.0,
        half_life=half_life,
        persistence=0.0,
        mean=0.0,
        variance=1.0,
        std=1.0,
        portmanteau=portmanteau,
    )

    spread = Spread(
        tickers=["A", "B"],
        beta=np.array([1.0, -1.0]),
        beta_index=0,
        values=pd.Series(dtype=float),
    )

    return CointegrationAnalysis(
        tickers=["A", "B"],
        rank=1,
        beta_index=0,
        beta=np.array([1.0, -1.0]),
        spread=spread,
        evaluation=evaluation,
    )


def test_select_filters_by_adf_pvalue() -> None:
    valid = make_analysis(
        half_life=1.0,
        portmanteau=1.0,
    )
    invalid = make_analysis(
        half_life=2.0,
        portmanteau=1.0,
        adf_pvalue=0.05,
    )

    selected = PortfolioSelector().select(
        [valid, invalid],
        top_n=10,
    )

    assert selected == [valid]


def test_select_filters_by_kpss_pvalue() -> None:
    valid = make_analysis(
        half_life=1.0,
        portmanteau=1.0,
    )
    invalid = make_analysis(
        half_life=2.0,
        portmanteau=1.0,
        kpss_pvalue=0.05,
    )

    selected = PortfolioSelector().select(
        [valid, invalid],
        top_n=10,
    )

    assert selected == [valid]


def test_select_filters_nonfinite_half_life() -> None:
    valid = make_analysis(
        half_life=1.0,
        portmanteau=1.0,
    )
    infinite = make_analysis(
        half_life=np.inf,
        portmanteau=2.0,
    )
    nan = make_analysis(
        half_life=np.nan,
        portmanteau=3.0,
    )

    selected = PortfolioSelector().select(
        [valid, infinite, nan],
        top_n=10,
    )

    assert selected == [valid]


def test_select_filters_nonfinite_portmanteau() -> None:
    valid = make_analysis(
        half_life=1.0,
        portmanteau=1.0,
    )
    infinite = make_analysis(
        half_life=2.0,
        portmanteau=np.inf,
    )
    nan = make_analysis(
        half_life=3.0,
        portmanteau=np.nan,
    )

    selected = PortfolioSelector().select(
        [valid, infinite, nan],
        top_n=10,
    )

    assert selected == [valid]


def test_select_sorts_by_half_life_then_portmanteau() -> None:
    half_life_2_low = make_analysis(
        half_life=2.0,
        portmanteau=1.0,
    )
    half_life_1 = make_analysis(
        half_life=1.0,
        portmanteau=1.0,
    )
    half_life_2_high = make_analysis(
        half_life=2.0,
        portmanteau=3.0,
    )

    selected = PortfolioSelector().select(
        [
            half_life_2_low,
            half_life_1,
            half_life_2_high,
        ],
        top_n=3,
    )

    assert selected == [
        half_life_1,
        half_life_2_high,
        half_life_2_low,
    ]


def test_select_limits_to_top_n() -> None:
    first = make_analysis(
        half_life=1.0,
        portmanteau=1.0,
    )
    second = make_analysis(
        half_life=2.0,
        portmanteau=1.0,
    )
    third = make_analysis(
        half_life=3.0,
        portmanteau=1.0,
    )

    selected = PortfolioSelector().select(
        [first, second, third],
        top_n=2,
    )

    assert selected == [
        first,
        second,
    ]


def test_select_rejects_nonpositive_top_n() -> None:
    with pytest.raises(
        ValueError,
        match="top_n must be positive",
    ):
        PortfolioSelector().select(
            [],
            top_n=0,
        )
