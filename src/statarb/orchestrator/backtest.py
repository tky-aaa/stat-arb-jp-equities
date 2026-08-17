from dataclasses import dataclass

from statarb.config.config import BacktestConfig
from statarb.core.backtest.api import BacktestAPI
from statarb.core.cointegration.api import CointegrationAPI
from statarb.core.data.api import DataAPI
from statarb.core.portfolio.api import PortfolioAPI
from statarb.core.report.api import ReportAPI
from statarb.core.screening.api import ScreeningAPI
from statarb.core.signal.api import SignalAPI


@dataclass(frozen=True)
class BacktestOrchestrator:
    config: BacktestConfig
    data_api: DataAPI | None = None
    screening_api: ScreeningAPI | None = None
    cointegration_api: CointegrationAPI | None = None
    portfolio_api: PortfolioAPI | None = None
    backtest_api: BacktestAPI | None = None
    report_api: ReportAPI | None = None

    def __post_init__(self) -> None:
        if self.data_api is None:
            object.__setattr__(
                self,
                "data_api",
                DataAPI(
                    self.config.universe,
                    self.config.training_start,
                    self.config.training_end,
                    self.config.test_start,
                    self.config.test_end,
                ),
            )

        if self.screening_api is None:
            object.__setattr__(
                self,
                "screening_api",
                ScreeningAPI(
                    self.config.screening_method,
                    self.config.n_clusters,
                    self.config.min_cluster_size,
                    self.config.max_cluster_size,
                    self.config.min_assets,
                    self.config.max_assets,
                    self.config.pca_components,
                ),
            )

        if self.cointegration_api is None:
            object.__setattr__(
                self,
                "cointegration_api",
                CointegrationAPI(
                    self.config.persistence_days,
                ),
            )

        if self.portfolio_api is None:
            signal_api = SignalAPI(
                self.config.zscore_window,
                self.config.entry_threshold,
                self.config.exit_threshold,
                self.config.spread_method,
                self.config.threshold_method,
            )

            object.__setattr__(
                self,
                "portfolio_api",
                PortfolioAPI(
                    signal_api=signal_api,
                    top_n_spreads=self.config.top_n_spreads,
                ),
            )

        if self.backtest_api is None:
            object.__setattr__(
                self,
                "backtest_api",
                BacktestAPI(
                    self.config.initial_capital,
                ),
            )

        if self.report_api is None:
            object.__setattr__(
                self,
                "report_api",
                ReportAPI(
                    results_path=self.config.results_path,
                ),
            )

    def run(self) -> None:
        print("[1/6] DataAPI: loading prices...")

        prices = self.data_api.service()

        print("[1/6] DataAPI: done")

        print("[2/6] ScreeningAPI: screening...")

        subgroups = self.screening_api.service(
            prices,
        )

        print(
            f"[2/6] ScreeningAPI: done ({len(subgroups)} subgroups)",
        )

        print("[3/6] CointegrationAPI: analyzing...")

        analyses = self.cointegration_api.service(
            prices,
            subgroups,
        )

        print(f"[3/6] CointegrationAPI: done ({len(analyses)} analyses)")

        print("[4/6] PortfolioAPI: generating portfolio...")

        decision = self.portfolio_api.service(
            prices,
            analyses,
        )

        print(f"[4/6] PortfolioAPI: done ({len(decision.signals)} signals)")

        print("[5/6] BacktestAPI: calculating returns...")

        result = self.backtest_api.service(
            prices,
            decision,
        )

        print("[5/6] BacktestAPI: done")

        print("[6/6] ReportAPI: saving results...")

        self.report_api.service(
            {
                "config": self.config,
                "prices": prices,
                "screening": subgroups,
                "cointegration": analyses,
                "portfolio": decision,
                "backtest": result,
            },
        )

        print("[6/6] ReportAPI: done")
