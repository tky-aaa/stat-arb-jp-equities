from dataclasses import dataclass

import pandas as pd

from statarb.config.contract import Group, Prices, SubGroup
from statarb.core.screening.clustering import Clusterizer
from statarb.core.screening.ff3 import FF3Estimator
from statarb.core.screening.groups import GroupGenerator
from statarb.core.screening.pca import PCAComputer
from statarb.core.screening.subgroups import SubgroupGenerator


@dataclass(frozen=True)
class ScreeningAPI:
    screening_method: str
    n_clusters: int
    min_cluster_size: int
    max_cluster_size: int
    min_assets: int
    max_assets: int
    pca_components: int

    def service(
        self,
        prices: Prices,
        *,
        factors: pd.DataFrame | None = None,
        universe: list[str] | None = None,
    ) -> list[SubGroup]:
        if self.screening_method == "full":
            if universe is None:
                universe = prices.training.columns.tolist()

            groups = [
                Group(
                    tickers=universe,
                )
            ]

        elif self.screening_method == "pca":
            returns = prices.training.pct_change().dropna()

            pca = PCAComputer()
            features = pca.compute_features(
                returns,
                self.pca_components,
            )

            clusterizer = Clusterizer()
            clusters = clusterizer.cluster(
                features,
                self.n_clusters,
            )

            generator = GroupGenerator()
            groups = generator.generate(
                clusters,
                min_size=self.min_cluster_size,
                max_size=self.max_cluster_size,
            )

        elif self.screening_method == "ff3":
            if factors is None:
                raise ValueError("factors are required for ff3 screening.")

            returns = prices.training.pct_change().dropna()

            estimator = FF3Estimator()
            features = estimator.estimate_exposure(
                returns,
                factors,
            )

            clusterizer = Clusterizer()
            clusters = clusterizer.cluster(
                features[
                    [
                        "beta_mkt",
                        "beta_smb",
                        "beta_hml",
                    ]
                ],
                self.n_clusters,
                scale=True,
            )

            generator = GroupGenerator()
            groups = generator.generate(
                clusters,
                min_size=self.min_cluster_size,
                max_size=self.max_cluster_size,
            )

        else:
            raise ValueError(f"Unknown screening method: {self.screening_method}")

        generator = SubgroupGenerator()

        return generator.generate(
            groups,
            min_assets=self.min_assets,
            max_assets=self.max_assets,
        )
