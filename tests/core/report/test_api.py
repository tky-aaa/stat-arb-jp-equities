from pathlib import Path

import pandas as pd
import pytest

from statarb.config.config import BacktestConfig
from statarb.config.contract import (
    BacktestResult,
    CointegrationAnalysis,
    Spread,
    SpreadEvaluation,
    SubGroup,
)
from statarb.core.report.api import ReportAPI


def test_service_saves_outputs(tmp_path: Path) -> None:
    analysis = CointegrationAnalysis(
        rank=1,
        beta_index=0,
        spread=Spread(
            tickers=["A", "B"],
            beta=pd.DataFrame(
                [[1.0, -1.0], [1.0, -1.0]],
                index=pd.date_range("2026-01-01", periods=2),
                columns=["A", "B"],
            ),
            intercept=pd.Series(
                [0.0, 0.0],
                index=pd.date_range("2026-01-01", periods=2),
                name="intercept",
            ),
            values=pd.Series(
                [0.0, 0.1],
                index=pd.date_range("2026-01-01", periods=2),
            ),
        ),
        evaluation=SpreadEvaluation(
            adf_stat=-2.0,
            adf_pvalue=0.05,
            kpss_stat=0.1,
            kpss_pvalue=0.1,
            rho1=0.5,
            phi=0.5,
            half_life=1.0,
            persistence=0.5,
            mean=0.0,
            variance=1.0,
            std=1.0,
            portmanteau=0.1,
        ),
    )

    outputs = {
        "screening": [
            SubGroup(tickers=["A", "B"]),
        ],
        "cointegration": [analysis],
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

    assert (experiment_path / "screening.csv").exists()
    assert (experiment_path / "cointegration.csv").exists()
    assert (experiment_path / "backtest.csv").exists()

    assert not (experiment_path / "screening.pkl").exists()
    assert not (experiment_path / "cointegration.pkl").exists()
    assert not (experiment_path / "backtest.pkl").exists()


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

    assert (experiment_path / "checkpoint" / "config.pkl").exists()
    assert not (experiment_path / "config.pkl").exists()

    loaded = ReportAPI(
        results_path=str(tmp_path),
    ).storage.load(
        experiment_path / "checkpoint" / "config.pkl",
    )

    assert loaded == config
