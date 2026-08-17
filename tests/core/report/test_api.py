from pathlib import Path

import pandas as pd
import pytest

from statarb.config.config import BacktestConfig
from statarb.config.contract import (
    BacktestResult,
    SubGroup,
)
from statarb.core.report.api import ReportAPI


def test_service_saves_outputs(tmp_path: Path) -> None:
    outputs = {
        "screening": [
            SubGroup(tickers=["A", "B"]),
        ],
        "cointegration": {"ticker": "A"},
        "backtest": BacktestResult(
            pnl=pd.Series([0.0, 0.1]),
            cumulative_pnl=pd.Series([0.0, 0.1]),
            equity=pd.Series([1.0, 1.1]),
        ),
    }

    experiment_path = ReportAPI(
        results_path=str(tmp_path),
    ).service(outputs)

    assert experiment_path.parent == tmp_path
    assert experiment_path.is_dir()

    assert (experiment_path / "screening.pkl").exists()
    assert (experiment_path / "cointegration.pkl").exists()
    assert (experiment_path / "backtest.pkl").exists()


def test_service_preserves_outputs(tmp_path: Path) -> None:
    outputs = {
        "screening": [
            SubGroup(tickers=["A", "B"]),
        ],
        "backtest": BacktestResult(
            pnl=pd.Series([0.0, 0.1]),
            cumulative_pnl=pd.Series([0.0, 0.1]),
            equity=pd.Series([1.0, 1.1]),
        ),
    }

    api = ReportAPI(
        results_path=str(tmp_path),
    )

    experiment_path = api.service(outputs)

    screening = api.storage.load(
        experiment_path / "screening.pkl",
    )
    backtest = api.storage.load(
        experiment_path / "backtest.pkl",
    )

    assert screening == [
        SubGroup(tickers=["A", "B"]),
    ]
    pd.testing.assert_series_equal(
        backtest.pnl,
        outputs["backtest"].pnl,
    )
    pd.testing.assert_series_equal(
        backtest.cumulative_pnl,
        outputs["backtest"].cumulative_pnl,
    )
    pd.testing.assert_series_equal(
        backtest.equity,
        outputs["backtest"].equity,
    )


def test_service_rejects_empty_outputs(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="outputs must not be empty",
    ):
        ReportAPI(
            results_path=str(tmp_path),
        ).service({})


def test_create_experiment_path(tmp_path: Path) -> None:
    api = ReportAPI(
        results_path=str(tmp_path),
    )

    experiment_path = api.create_experiment_path()

    assert experiment_path.parent == tmp_path
    assert experiment_path.is_dir()


def test_save_checkpoint(tmp_path: Path) -> None:
    api = ReportAPI(
        results_path=str(tmp_path),
    )

    experiment_path = api.create_experiment_path()

    api.save_checkpoint(
        experiment_path,
        "prices",
        {"value": 1},
    )

    checkpoint = experiment_path / "checkpoint" / "prices.pkl"

    assert checkpoint.exists()

    loaded = api.storage.load(checkpoint)

    assert loaded == {"value": 1}


def test_service_saves_config(tmp_path: Path) -> None:
    config = BacktestConfig()

    experiment_path = ReportAPI(
        results_path=str(tmp_path),
    ).service(
        {
            "config": config,
        },
    )

    assert (experiment_path / "config.pkl").exists()

    loaded = ReportAPI(
        results_path=str(tmp_path),
    ).storage.load(
        experiment_path / "config.pkl",
    )

    assert loaded == config
