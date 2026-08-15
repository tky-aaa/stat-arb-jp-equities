from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from newstatarb.config.config import LiveConfig
from newstatarb.config.contract import (
    CointegrationAnalysis,
    PortfolioDecision,
    Prices,
    Signal,
)
from newstatarb.orchestrator.live_short import LiveShortOrchestrator


def test_live_short_loads_strategy_updates_portfolio_and_executes(
    tmp_path,
) -> None:
    strategy_path = tmp_path / "strategy.pkl"

    config = LiveConfig(
        strategy_path=str(strategy_path),
    )

    analysis = Mock(spec=CointegrationAnalysis)

    strategy = Mock()
    strategy.training_start = config.training_start
    strategy.training_end = config.training_end
    strategy.analyses = [analysis]

    storage = Mock()
    storage.load.return_value = strategy

    prices = Prices(
        training=pd.DataFrame(),
        test=pd.DataFrame(),
    )

    signal = Signal(
        spread=pd.Series([1.0]),
        zscore=pd.Series([0.0]),
        position=pd.Series([1.0]),
    )

    decision = PortfolioDecision(
        analyses=[analysis],
        signals=[signal],
        weights=[1.0],
    )

    portfolio_api = Mock()
    portfolio_api.service.return_value = decision

    execution = Mock()

    execution_builder = Mock()
    execution_builder.build.return_value = execution

    ibkr_client = Mock()
    ibkr_client.execute.return_value = [{"status": "Filled"}]

    data_api = Mock()
    data_api.service.return_value = prices

    orchestrator = LiveShortOrchestrator(
        config=config,
        portfolio_api=portfolio_api,
        storage=storage,
        data_api=data_api,
        execution_builder=execution_builder,
        ibkr_client=ibkr_client,
    )

    result = orchestrator.run(
        test_start="2026-01-01",
        test_end="2026-01-02",
    )

    storage.load.assert_called_once_with(
        Path(config.strategy_path),
    )

    data_api.service.assert_called_once_with()

    portfolio_api.service.assert_called_once_with(
        prices,
        strategy.analyses,
    )

    execution_builder.build.assert_called_once_with(
        decision,
    )

    ibkr_client.connect.assert_called_once_with()
    ibkr_client.execute.assert_called_once_with(execution)
    ibkr_client.close.assert_called_once_with()

    assert result == [{"status": "Filled"}]
