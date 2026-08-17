from pathlib import Path
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


def test_init_builds_apis_from_config() -> None:
    config = BacktestConfig(
        universe="topix10",
        training_start="2025-02-01",
        training_end="2025-11-30",
        test_start="2026-02-01",
        test_end="2026-06-30",
        screening_method="pca",
        n_clusters=7,
        min_cluster_size=4,
        max_cluster_size=15,
        min_assets=2,
        max_assets=3,
        pca_components=4,
        persistence_days=45,
        zscore_window=50,
        entry_threshold=2.5,
        exit_threshold=0.4,
        threshold_method="empirical",
        spread_method="kalman",
        top_n_spreads=3,
        results_path="test_results",
        initial_capital=2.0,
    )

    orchestrator = BacktestOrchestrator(config)

    assert orchestrator.data_api.universe == config.universe
    assert orchestrator.data_api.training_start == config.training_start
    assert orchestrator.data_api.training_end == config.training_end
    assert orchestrator.data_api.test_start == config.test_start
    assert orchestrator.data_api.test_end == config.test_end

    assert orchestrator.screening_api.screening_method == config.screening_method
    assert orchestrator.screening_api.n_clusters == config.n_clusters
    assert orchestrator.screening_api.min_cluster_size == config.min_cluster_size
    assert orchestrator.screening_api.max_cluster_size == config.max_cluster_size
    assert orchestrator.screening_api.min_assets == config.min_assets
    assert orchestrator.screening_api.max_assets == config.max_assets
    assert orchestrator.screening_api.pca_components == config.pca_components

    assert orchestrator.cointegration_api.persistence_days == config.persistence_days

    assert orchestrator.portfolio_api.top_n_spreads == config.top_n_spreads
    assert orchestrator.portfolio_api.signal_api.zscore_window == config.zscore_window
    assert (
        orchestrator.portfolio_api.signal_api.entry_threshold == config.entry_threshold
    )
    assert orchestrator.portfolio_api.signal_api.exit_threshold == config.exit_threshold
    assert (
        orchestrator.portfolio_api.signal_api.threshold_method
        == config.threshold_method
    )
    assert orchestrator.portfolio_api.signal_api.spread_method == config.spread_method

    assert orchestrator.backtest_api.initial_capital == config.initial_capital

    assert orchestrator.report_api.results_path == Path(config.results_path)
