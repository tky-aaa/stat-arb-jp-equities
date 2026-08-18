from dataclasses import asdict, fields
from pathlib import Path

import pandas as pd

from statarb.config.config import BacktestConfig
from statarb.config.contract import (
    BacktestMetrics,
    BacktestResult,
    CointegrationAnalysis,
    PortfolioDecision,
    Prices,
    SubGroup,
)


class ReportExporter:
    def export(
        self,
        output_path: Path,
        outputs: dict[str, object],
    ) -> None:
        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        for name, output in outputs.items():
            if name == "config":
                self._export_config(
                    output_path,
                    output,
                )

            elif name == "prices":
                self._export_prices(
                    output_path,
                    output,
                )

            elif name == "screening":
                self._export_screening(
                    output_path,
                    output,
                )

            elif name == "cointegration":
                self._export_cointegration(
                    output_path,
                    output,
                )

            elif name == "portfolio":
                self._export_portfolio(
                    output_path,
                    output,
                )

            elif name == "backtest":
                self._export_backtest(
                    output_path,
                    output,
                )

            elif name == "metrics":
                self._export_metrics(
                    output_path,
                    output,
                )

    def _export_config(
        self,
        output_path: Path,
        config: object,
    ) -> None:
        if not isinstance(config, BacktestConfig):
            raise TypeError("config must be a BacktestConfig.")

        rows = [
            {
                "parameter": field.name,
                "value": getattr(config, field.name),
            }
            for field in fields(config)
        ]

        pd.DataFrame(rows).to_csv(
            output_path / "config.csv",
            index=False,
        )

    def _export_prices(
        self,
        output_path: Path,
        prices: object,
    ) -> None:
        if not isinstance(prices, Prices):
            raise TypeError("prices must be a Prices.")

        prices.training.to_csv(
            output_path / "prices_training.csv",
        )

        prices.test.to_csv(
            output_path / "prices_test.csv",
        )

    def _export_screening(
        self,
        output_path: Path,
        subgroups: object,
    ) -> None:
        if not isinstance(subgroups, list):
            raise TypeError("screening must be a list of SubGroup.")

        if not all(isinstance(subgroup, SubGroup) for subgroup in subgroups):
            raise TypeError("screening must contain only SubGroup objects.")

        frame = pd.DataFrame(
            {
                "tickers": ["_".join(subgroup.tickers) for subgroup in subgroups],
            },
        )

        frame.to_csv(
            output_path / "screening.csv",
            index=False,
        )

    def _export_cointegration(
        self,
        output_path: Path,
        analyses: object,
    ) -> None:
        if not isinstance(analyses, list):
            raise TypeError(
                "cointegration must be a list of CointegrationAnalysis.",
            )

        if not all(
            isinstance(analysis, CointegrationAnalysis) for analysis in analyses
        ):
            raise TypeError(
                "cointegration must contain only CointegrationAnalysis objects.",
            )

        rows: list[dict[str, object]] = []

        for analysis in analyses:
            evaluation = analysis.evaluation

            for timestamp, spread in analysis.spread.values.items():
                rows.append(
                    {
                        "timestamp": timestamp,
                        "tickers": "_".join(analysis.spread.tickers),
                        "rank": analysis.rank,
                        "beta_index": analysis.beta_index,
                        "beta": str(analysis.spread.beta.iloc[0].to_numpy().tolist()),
                        "adf_stat": evaluation.adf_stat,
                        "adf_pvalue": evaluation.adf_pvalue,
                        "kpss_stat": evaluation.kpss_stat,
                        "kpss_pvalue": evaluation.kpss_pvalue,
                        "rho1": evaluation.rho1,
                        "phi": evaluation.phi,
                        "half_life": evaluation.half_life,
                        "persistence": evaluation.persistence,
                        "mean": evaluation.mean,
                        "variance": evaluation.variance,
                        "std": evaluation.std,
                        "portmanteau": evaluation.portmanteau,
                        "spread": spread,
                    },
                )

        frame = pd.DataFrame(rows)

        frame.to_csv(
            output_path / "cointegration.csv",
            index=False,
        )

    def _export_portfolio(
        self,
        output_path: Path,
        portfolio: object,
    ) -> None:
        if not isinstance(portfolio, PortfolioDecision):
            raise TypeError(
                "portfolio must be a PortfolioDecision.",
            )

        if not (
            len(portfolio.analyses) == len(portfolio.signals) == len(portfolio.weights)
        ):
            raise ValueError(
                "portfolio analyses, signals, and weights must have the same length.",
            )

        rows: list[dict[str, object]] = []

        for analysis, signal, weight in zip(
            portfolio.analyses,
            portfolio.signals,
            portfolio.weights,
        ):
            for timestamp in signal.position.index:
                rows.append(
                    {
                        "timestamp": timestamp,
                        "tickers": "_".join(analysis.spread.tickers),
                        "weight": weight,
                        "spread": signal.spread.loc[timestamp],
                        "zscore": signal.zscore.loc[timestamp],
                        "position": signal.position.loc[timestamp],
                    },
                )

        frame = pd.DataFrame(rows)

        frame.to_csv(
            output_path / "portfolio.csv",
            index=False,
        )

    def _export_backtest(
        self,
        output_path: Path,
        result: object,
    ) -> None:
        if not isinstance(result, BacktestResult):
            raise TypeError("backtest must be a BacktestResult.")

        frame = pd.DataFrame(
            {
                "pnl": result.pnl,
                "cumulative_pnl": result.cumulative_pnl,
                "equity": result.equity,
            },
        )

        frame.to_csv(
            output_path / "backtest.csv",
        )

    def _export_metrics(
        self,
        output_path: Path,
        metrics: object,
    ) -> None:
        if not isinstance(metrics, BacktestMetrics):
            raise TypeError("metrics must be a BacktestMetrics.")

        pd.DataFrame(
            [asdict(metrics)],
        ).to_csv(
            output_path / "metrics.csv",
            index=False,
        )
