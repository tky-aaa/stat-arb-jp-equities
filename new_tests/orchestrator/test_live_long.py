from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from newstatarb.config.config import LiveConfig
from newstatarb.config.contract import (
    CointegrationAnalysis,
    LiveStrategy,
    Prices,
    SubGroup,
)
from newstatarb.orchestrator.live_long import LiveLongOrchestrator


def test_run_creates_and_saves_strategy() -> None:
    config = LiveConfig(
        strategy_path="data/live/test_strategy.pkl",
    )

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

    data_api = Mock()
    screening_api = Mock()
    cointegration_api = Mock()
    storage = Mock()

    data_api.service.return_value = prices
    screening_api.service.return_value = subgroups
    cointegration_api.service.return_value = analyses

    orchestrator = LiveLongOrchestrator(
        config=config,
        data_api=data_api,
        screening_api=screening_api,
        cointegration_api=cointegration_api,
        storage=storage,
    )

    orchestrator.run()

    data_api.service.assert_called_once_with()

    screening_api.service.assert_called_once_with(
        prices,
    )

    cointegration_api.service.assert_called_once_with(
        prices,
        subgroups,
    )

    storage.save.assert_called_once()

    saved_strategy = storage.save.call_args.args[0]
    saved_path = storage.save.call_args.args[1]

    assert isinstance(saved_strategy, LiveStrategy)
    assert saved_strategy.analyses == analyses
    assert saved_strategy.training_start == config.training_start
    assert saved_strategy.training_end == config.training_end
    assert saved_path == Path(config.strategy_path)
