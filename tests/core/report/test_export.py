from pathlib import Path

import pandas as pd

from statarb.config.config import BacktestConfig
from statarb.config.contract import (
    BacktestMetrics,
    BacktestResult,
    CointegrationAnalysis,
    PortfolioDecision,
    Prices,
    Signal,
    Spread,
    SpreadEvaluation,
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


def test_export_cointegration(tmp_path: Path) -> None:
    exporter = ReportExporter()

    spread = Spread(
        tickers=["A", "B"],
        beta=pd.Series([1.0, -0.5]).to_numpy(),
        beta_index=0,
        values=pd.Series(
            [0.1, 0.2],
            index=pd.date_range("2026-01-01", periods=2),
        ),
    )

    evaluation = SpreadEvaluation(
        adf_stat=-3.0,
        adf_pvalue=0.01,
        kpss_stat=0.2,
        kpss_pvalue=0.1,
        rho1=0.8,
        phi=0.8,
        half_life=3.1,
        persistence=0.8,
        mean=0.15,
        variance=0.0025,
        std=0.05,
        portmanteau=2.0,
    )

    analysis = CointegrationAnalysis(
        tickers=["A", "B"],
        rank=1,
        beta_index=0,
        beta=pd.Series([1.0, -0.5]).to_numpy(),
        spread=spread,
        evaluation=evaluation,
    )

    exporter.export(
        tmp_path,
        {"cointegration": [analysis]},
    )

    frame = pd.read_csv(
        tmp_path / "cointegration.csv",
    )

    assert list(frame.columns) == [
        "timestamp",
        "tickers",
        "rank",
        "beta_index",
        "beta",
        "adf_stat",
        "adf_pvalue",
        "kpss_stat",
        "kpss_pvalue",
        "rho1",
        "phi",
        "half_life",
        "persistence",
        "mean",
        "variance",
        "std",
        "portmanteau",
        "spread",
    ]

    assert len(frame) == 2
    assert frame["tickers"].tolist() == ["A_B", "A_B"]
    assert frame["rank"].tolist() == [1, 1]
    assert frame["spread"].tolist() == [0.1, 0.2]


def test_export_portfolio(tmp_path: Path) -> None:
    exporter = ReportExporter()

    spread = Spread(
        tickers=["A", "B"],
        beta=pd.Series([1.0, -0.5]).to_numpy(),
        beta_index=0,
        values=pd.Series(
            [0.1, 0.2],
            index=pd.date_range("2026-01-01", periods=2),
        ),
    )

    evaluation = SpreadEvaluation(
        adf_stat=-3.0,
        adf_pvalue=0.01,
        kpss_stat=0.2,
        kpss_pvalue=0.1,
        rho1=0.8,
        phi=0.8,
        half_life=3.1,
        persistence=0.8,
        mean=0.15,
        variance=0.0025,
        std=0.05,
        portmanteau=2.0,
    )

    analysis = CointegrationAnalysis(
        tickers=["A", "B"],
        rank=1,
        beta_index=0,
        beta=pd.Series([1.0, -0.5]).to_numpy(),
        spread=spread,
        evaluation=evaluation,
    )

    signal = Signal(
        spread=pd.Series(
            [0.1, 0.2],
            index=pd.date_range("2026-01-01", periods=2),
        ),
        zscore=pd.Series(
            [1.0, 2.0],
            index=pd.date_range("2026-01-01", periods=2),
        ),
        position=pd.Series(
            [0.0, -1.0],
            index=pd.date_range("2026-01-01", periods=2),
        ),
    )

    portfolio = PortfolioDecision(
        analyses=[analysis],
        signals=[signal],
        weights=[0.1],
    )

    exporter.export(
        tmp_path,
        {"portfolio": portfolio},
    )

    frame = pd.read_csv(
        tmp_path / "portfolio.csv",
    )

    assert list(frame.columns) == [
        "timestamp",
        "tickers",
        "weight",
        "spread",
        "zscore",
        "position",
    ]

    assert len(frame) == 2
    assert frame["tickers"].tolist() == ["A_B", "A_B"]
    assert frame["weight"].tolist() == [0.1, 0.1]
    assert frame["zscore"].tolist() == [1.0, 2.0]
    assert frame["position"].tolist() == [0.0, -1.0]


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
