import numpy as np
import pandas as pd
import pytest

from statarb.config.contract import (
    CointegrationAnalysis,
    Spread,
    SpreadEvaluation,
)
from statarb.core.portfolio.allocator import PortfolioAllocator


def make_analysis(
    values: list[float],
) -> CointegrationAnalysis:
    spread_values = pd.Series(
        values,
        dtype=float,
    )

    spread = Spread(
        tickers=["A", "B"],
        beta=np.array([1.0, -1.0]),
        beta_index=0,
        values=spread_values,
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
        portmanteau=0.0,
    )

    return CointegrationAnalysis(
        tickers=["A", "B"],
        rank=1,
        beta_index=0,
        beta=np.array([1.0, -1.0]),
        spread=spread,
        evaluation=evaluation,
    )


def test_allocate_returns_empty_for_empty_analyses() -> None:
    weights = PortfolioAllocator().allocate([])

    assert weights == []


def test_allocate_uses_inverse_volatility_weights() -> None:
    low_vol = make_analysis([1.0, 1.0, 1.0, 1.0, 2.0])
    high_vol = make_analysis([1.0, 1.0, 1.0, 1.0, 3.0])

    weights = PortfolioAllocator().allocate(
        [low_vol, high_vol],
        max_weight=1.0,
    )

    assert weights[0] > weights[1]
    assert sum(weights) == pytest.approx(1.0)


def test_allocate_applies_max_weight() -> None:
    low_vol = make_analysis([1.0, 1.0, 1.0, 1.0, 1.1])
    high_vol = make_analysis([1.0, 1.0, 1.0, 1.0, 3.0])

    weights = PortfolioAllocator().allocate(
        [low_vol, high_vol],
        max_weight=0.6,
    )

    assert weights[0] <= 0.6
    assert sum(weights) == pytest.approx(1.0)


def test_allocate_redistributes_capped_weight() -> None:
    first = make_analysis([1.0, 1.0, 1.0, 1.0, 1.1])
    second = make_analysis([1.0, 1.0, 1.0, 1.0, 2.0])
    third = make_analysis([1.0, 1.0, 1.0, 1.0, 3.0])

    weights = PortfolioAllocator().allocate(
        [first, second, third],
        max_weight=0.4,
    )

    assert all(weight <= 0.4 for weight in weights)
    assert sum(weights) == pytest.approx(1.0)


def test_allocate_rejects_nonpositive_max_weight() -> None:
    analysis = make_analysis([1.0, 2.0, 3.0])

    with pytest.raises(
        ValueError,
        match="max_weight must be positive",
    ):
        PortfolioAllocator().allocate(
            [analysis],
            max_weight=0.0,
        )


def test_allocate_rejects_empty_spread() -> None:
    analysis = make_analysis([])

    with pytest.raises(
        ValueError,
        match="Spread contains no valid observations",
    ):
        PortfolioAllocator().allocate([analysis])


def test_allocate_rejects_zero_volatility() -> None:
    analysis = make_analysis(
        [1.0, 1.0, 1.0, 1.0],
    )

    with pytest.raises(
        ValueError,
        match="Spread volatility must be positive and finite",
    ):
        PortfolioAllocator().allocate([analysis])
