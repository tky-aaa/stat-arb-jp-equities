from pathlib import Path

import pandas as pd

from statarb.config.config import BacktestConfig
from statarb.config.contract import (
    BacktestMetrics,
    BacktestResult,
    Prices,
    SubGroup,
)
from statarb.core.report.export import ReportExporter


def test_export_config(tmp_path: Path) -> None:
    exporter = ReportExporter()

    config = BacktestConfig()

    exporter.export(
        tmp_path,
        {"config": config},
    )

    output = tmp_path / "config.csv"

    assert output.exists()

    frame = pd.read_csv(output)

    assert set(frame["parameter"]) >= {
        "universe",
        "training_start",
        "test_start",
        "screening_method",
        "n_clusters",
        "top_n_spreads",
    }


def test_export_prices(tmp_path: Path) -> None:
    exporter = ReportExporter()

    prices = Prices(
        training=pd.DataFrame(
            {
                "A": [100.0, 101.0],
                "B": [200.0, 201.0],
            },
            index=pd.date_range("2026-01-01", periods=2),
        ),
        test=pd.DataFrame(
            {
                "A": [102.0, 103.0],
                "B": [202.0, 203.0],
            },
            index=pd.date_range("2026-01-03", periods=2),
        ),
    )

    exporter.export(
        tmp_path,
        {"prices": prices},
    )

    assert (tmp_path / "prices_training.csv").exists()
    assert (tmp_path / "prices_test.csv").exists()

    training = pd.read_csv(
        tmp_path / "prices_training.csv",
        index_col=0,
    )

    assert list(training.columns) == ["A", "B"]


def test_export_screening(tmp_path: Path) -> None:
    exporter = ReportExporter()

    subgroups = [
        SubGroup(tickers=["A", "B"]),
        SubGroup(tickers=["C", "D", "E"]),
    ]

    exporter.export(
        tmp_path,
        {"screening": subgroups},
    )

    frame = pd.read_csv(
        tmp_path / "screening.csv",
    )

    assert frame["tickers"].tolist() == [
        "A_B",
        "C_D_E",
    ]


def test_export_backtest(tmp_path: Path) -> None:
    exporter = ReportExporter()

    result = BacktestResult(
        pnl=pd.Series(
            [0.0, 0.1],
            index=pd.date_range("2026-01-01", periods=2),
        ),
        cumulative_pnl=pd.Series(
            [0.0, 0.1],
            index=pd.date_range("2026-01-01", periods=2),
        ),
        equity=pd.Series(
            [1.0, 1.1],
            index=pd.date_range("2026-01-01", periods=2),
        ),
    )

    exporter.export(
        tmp_path,
        {"backtest": result},
    )

    frame = pd.read_csv(
        tmp_path / "backtest.csv",
        index_col=0,
    )

    assert list(frame.columns) == [
        "pnl",
        "cumulative_pnl",
        "equity",
    ]

    assert frame["equity"].tolist() == [1.0, 1.1]


def test_export_metrics(tmp_path: Path) -> None:
    exporter = ReportExporter()

    metrics = BacktestMetrics(
        total_return=0.1,
        annualized_return=0.2,
        volatility=0.15,
        sharpe_ratio=1.3,
        max_drawdown=-0.05,
        win_rate=0.6,
        number_of_trades=10,
    )

    exporter.export(
        tmp_path,
        {"metrics": metrics},
    )

    frame = pd.read_csv(
        tmp_path / "metrics.csv",
    )

    assert frame.loc[0, "total_return"] == 0.1
    assert frame.loc[0, "sharpe_ratio"] == 1.3
    assert frame.loc[0, "number_of_trades"] == 10
