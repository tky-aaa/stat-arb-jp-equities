from dataclasses import dataclass
from pathlib import Path

from statarb.config.config import LiveConfig
from statarb.core.data.api import DataAPI
from statarb.core.live.execution import LiveExecutionBuilder
from statarb.core.live.ibkr import IBKRClient
from statarb.core.live.storage import LiveStrategyStorage
from statarb.core.portfolio.api import PortfolioAPI
from statarb.core.signal.api import SignalAPI


@dataclass(frozen=True)
class LiveShortOrchestrator:
    config: LiveConfig
    portfolio_api: PortfolioAPI | None = None
    storage: LiveStrategyStorage | None = None
    data_api: DataAPI | None = None
    execution_builder: LiveExecutionBuilder | None = None
    ibkr_client: IBKRClient | None = None

    def __post_init__(self) -> None:
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

        if self.storage is None:
            object.__setattr__(
                self,
                "storage",
                LiveStrategyStorage(),
            )

        if self.execution_builder is None:
            object.__setattr__(
                self,
                "execution_builder",
                LiveExecutionBuilder(),
            )

        if self.ibkr_client is None:
            object.__setattr__(
                self,
                "ibkr_client",
                IBKRClient(),
            )

    def run(
        self,
        test_start: str,
        test_end: str,
    ):
        print("[1/6] LiveStrategyStorage: loading strategy...")
        strategy = self.storage.load(
            Path(self.config.strategy_path),
        )
        print(f"[1/6] LiveStrategyStorage: done ({len(strategy.analyses)} analyses)")

        print("[2/6] DataAPI: loading prices...")
        if self.data_api is None:
            data_api = DataAPI(
                self.config.universe,
                strategy.training_start,
                strategy.training_end,
                test_start,
                test_end,
            )
        else:
            data_api = self.data_api

        prices = data_api.service()
        print("[2/6] DataAPI: done")

        print("[3/6] PortfolioAPI: generating portfolio...")
        decision = self.portfolio_api.service(
            prices,
            strategy.analyses,
        )
        print(f"[3/6] PortfolioAPI: done ({len(decision.signals)} signals)")

        print("[4/6] LiveExecutionBuilder: building execution...")
        execution = self.execution_builder.build(
            decision,
        )
        print("[4/6] LiveExecutionBuilder: done")

        print("[5/6] IBKRClient: connecting...")
        self.ibkr_client.connect()
        print("[5/6] IBKRClient: connected")

        try:
            print("[6/6] IBKRClient: executing orders...")
            return self.ibkr_client.execute(
                execution,
            )
        finally:
            print("[6/6] IBKRClient: closing connection...")
            self.ibkr_client.close()
            print("[6/6] IBKRClient: closed")
