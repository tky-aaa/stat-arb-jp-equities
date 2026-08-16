from unittest.mock import Mock

import numpy as np
import pandas as pd

from statarb.config.contract import (
    CointegrationAnalysis,
    Prices,
    Signal,
    Spread,
    SpreadEvaluation,
)
from statarb.core.portfolio.api import PortfolioAPI


def make_signal(
    values: list[float],
) -> Signal:
    index = pd.date_range(
        "2025-01-01",
        periods=len(values),
    )

    return Signal(
        spread=pd.Series(
            values,
            index=index,
        ),
        zscore=pd.Series(
            [0.0] * len(values),
            index=index,
        ),
        position=pd.Series(
            [0.0] * len(values),
            index=index,
        ),
    )


def make_analysis(
    *,
    half_life: float,
    portmanteau: float,
    spread_values: list[float],
) -> CointegrationAnalysis:
    evaluation = SpreadEvaluation(
        adf_stat=0.0,
        adf_pvalue=0.01,
        kpss_stat=0.0,
        kpss_pvalue=0.10,
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
        values=pd.Series(
            spread_values,
            dtype=float,
        ),
    )

    return CointegrationAnalysis(
        tickers=["A", "B"],
        rank=1,
        beta_index=0,
        beta=np.array([1.0, -1.0]),
        spread=spread,
        evaluation=evaluation,
    )


def make_api(
    signals: list[Signal],
    *,
    top_n_spreads: int = 2,
    max_weight: float = 0.2,
    max_leverage: float = 1.0,
) -> PortfolioAPI:
    signal_api = Mock()
    signal_api.service.return_value = signals

    return PortfolioAPI(
        signal_api=signal_api,
        top_n_spreads=top_n_spreads,
        max_weight=max_weight,
        max_leverage=max_leverage,
    )


def make_prices() -> Prices:
    return Prices(
        training=pd.DataFrame(),
        test=pd.DataFrame(),
    )


def test_portfolio_api_returns_portfolio_decision() -> None:
    analyses = [
        make_analysis(
            half_life=1.0,
            portmanteau=1.0,
            spread_values=[1.0, 1.1, 1.2],
        ),
        make_analysis(
            half_life=2.0,
            portmanteau=1.0,
            spread_values=[1.0, 1.2, 1.4],
        ),
    ]

    api = make_api(
        [
            make_signal([1.0, 1.1, 1.2]),
            make_signal([1.0, 1.2, 1.4]),
        ],
    )

    decision = api.service(
        make_prices(),
        analyses,
    )

    assert decision.analyses == analyses
    assert len(decision.weights) == 2


def test_portfolio_api_selects_top_n_spreads() -> None:
    first = make_analysis(
        half_life=1.0,
        portmanteau=1.0,
        spread_values=[1.0, 1.1, 1.2],
    )
    second = make_analysis(
        half_life=2.0,
        portmanteau=1.0,
        spread_values=[1.0, 1.2, 1.4],
    )
    third = make_analysis(
        half_life=3.0,
        portmanteau=1.0,
        spread_values=[1.0, 1.3, 1.6],
    )

    api = make_api(
        [
            make_signal([1.0, 1.1, 1.2]),
            make_signal([1.0, 1.2, 1.4]),
            make_signal([1.0, 1.3, 1.6]),
        ],
    )

    decision = api.service(
        make_prices(),
        [third, first, second],
    )

    assert decision.analyses == [
        first,
        second,
    ]
    assert len(decision.weights) == 2


def test_portfolio_api_respects_max_weight() -> None:
    first = make_analysis(
        half_life=1.0,
        portmanteau=1.0,
        spread_values=[1.0, 1.01, 1.02],
    )
    second = make_analysis(
        half_life=2.0,
        portmanteau=1.0,
        spread_values=[1.0, 2.0, 3.0],
    )

    api = make_api(
        [
            make_signal([1.0, 1.01, 1.02]),
            make_signal([1.0, 2.0, 3.0]),
        ],
        max_weight=0.6,
    )
    decision = api.service(
        prices=Prices(
            training=pd.DataFrame(),
            test=pd.DataFrame(),
        ),
        analyses=[first, second],
    )

    assert all(abs(weight) <= 0.6 for weight in decision.weights)


def test_portfolio_api_respects_max_leverage() -> None:
    analyses = [
        make_analysis(
            half_life=1.0,
            portmanteau=1.0,
            spread_values=[1.0, 1.1, 1.2],
        ),
        make_analysis(
            half_life=2.0,
            portmanteau=1.0,
            spread_values=[1.0, 1.2, 1.4],
        ),
    ]

    api = make_api(
        [
            make_signal([1.0, 1.1, 1.2]),
            make_signal([1.0, 1.2, 1.4]),
        ],
        max_leverage=0.5,
    )

    decision = api.service(
        prices=Prices(
            training=pd.DataFrame(),
            test=pd.DataFrame(),
        ),
        analyses=analyses,
    )

    gross_leverage = sum(abs(weight) for weight in decision.weights)

    assert gross_leverage <= 0.5
