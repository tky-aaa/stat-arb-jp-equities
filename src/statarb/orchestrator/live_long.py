from dataclasses import dataclass
from pathlib import Path

from statarb.config.config import LiveConfig
from statarb.config.contract import LiveStrategy
from statarb.core.cointegration.api import CointegrationAPI
from statarb.core.data.api import DataAPI
from statarb.core.live.storage import LiveStrategyStorage
from statarb.core.screening.api import ScreeningAPI


@dataclass(frozen=True)
class LiveLongOrchestrator:
    config: LiveConfig
    data_api: DataAPI | None = None
    screening_api: ScreeningAPI | None = None
    cointegration_api: CointegrationAPI | None = None
    storage: LiveStrategyStorage | None = None

    def __post_init__(self) -> None:
        if self.data_api is None:
            object.__setattr__(
                self,
                "data_api",
                DataAPI(
                    self.config.universe,
                    self.config.training_start,
                    self.config.training_end,
                    self.config.training_start,
                    self.config.training_end,
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

        if self.storage is None:
            object.__setattr__(
                self,
                "storage",
                LiveStrategyStorage(),
            )

    def run(self) -> None:
        prices = self.data_api.service()

        subgroups = self.screening_api.service(
            prices,
        )

        analyses = self.cointegration_api.service(
            prices,
            subgroups,
        )

        strategy = LiveStrategy(
            analyses=analyses,
            training_start=self.config.training_start,
            training_end=self.config.training_end,
        )

        self.storage.save(
            strategy,
            Path(self.config.strategy_path),
        )
