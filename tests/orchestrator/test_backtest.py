from unittest.mock import Mock, call

import pandas as pd

from statarb.config.config import BacktestConfig
from statarb.config.contract import (
    BacktestResult,
    CointegrationAnalysis,
    PortfolioDecision,
    Prices,
    Signal,
    SubGroup,
)
from statarb.orchestrator.backtest import BacktestOrchestrator


def test_run_passes_contracts_through_pipeline() -> None:
    config = BacktestConfig()

    prices = Prices(
        training=pd.DataFrame(),
        test=pd.DataFrame(),
    )

    subgroups = [
        SubGroup(tickers=["A", "B"]),
    ]

    analyses = [
        Mock(spec=CointegrationAnalysis),
    ]

    signals = [
        Signal(
            spread=pd.Series([1.0]),
            zscore=pd.Series([0.0]),
            position=pd.Series([0.0]),
        ),
    ]

    decision = PortfolioDecision(
        analyses=analyses,
        signals=signals,
        weights=[1.0],
    )

    result = BacktestResult(
        pnl=pd.Series([0.0]),
        cumulative_pnl=pd.Series([0.0]),
        equity=pd.Series([1.0]),
    )

    data_api = Mock()
    screening_api = Mock()
    cointegration_api = Mock()
    portfolio_api = Mock()
    backtest_api = Mock()
    report_api = Mock()

    experiment_path = Mock()
    report_api.create_experiment_path.return_value = experiment_path

    data_api.service.return_value = prices
    screening_api.service.return_value = subgroups
    cointegration_api.service.return_value = analyses
    portfolio_api.service.return_value = decision
    backtest_api.service.return_value = result

    orchestrator = BacktestOrchestrator(
        config=config,
        data_api=data_api,
        screening_api=screening_api,
        cointegration_api=cointegration_api,
        portfolio_api=portfolio_api,
        backtest_api=backtest_api,
        report_api=report_api,
    )

    orchestrator.run()

    data_api.service.assert_called_once_with()
    screening_api.service.assert_called_once_with(prices)
    cointegration_api.service.assert_called_once_with(
        prices,
        subgroups,
    )
    portfolio_api.service.assert_called_once_with(
        prices,
        analyses,
    )
    backtest_api.service.assert_called_once_with(
        prices,
        decision,
    )

    report_api.create_experiment_path.assert_called_once_with()

    assert report_api.save_checkpoint.call_args_list == [
        call(
            experiment_path,
            "prices",
            prices,
        ),
        call(
            experiment_path,
            "screening",
            subgroups,
        ),
        call(
            experiment_path,
            "cointegration",
            analyses,
        ),
        call(
            experiment_path,
            "portfolio",
            decision,
        ),
        call(
            experiment_path,
            "backtest",
            result,
        ),
    ]

    report_api.service.assert_called_once_with(
        {
            "config": config,
            "prices": prices,
            "screening": subgroups,
            "cointegration": analyses,
            "portfolio": decision,
            "backtest": result,
        },
        experiment_path=experiment_path,
    )
