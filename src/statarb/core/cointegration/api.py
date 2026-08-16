from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from statarb.config.contract import (
    CointegrationAnalysis,
    Prices,
    SubGroup,
)
from statarb.core.cointegration.worker import CointegrationWorker


@dataclass(frozen=True)
class CointegrationAPI:
    persistence_days: int

    def service(
        self,
        prices: Prices,
        subgroups: list[SubGroup],
    ) -> list[CointegrationAnalysis]:
        log_prices = np.log(prices.training)

        if not subgroups:
            return []

        batch_size = 1_000

        workers = [
            CointegrationWorker(
                log_prices=log_prices,
                subgroups=subgroups[start : start + batch_size],
                persistence_days=self.persistence_days,
            )
            for start in range(
                0,
                len(subgroups),
                batch_size,
            )
        ]

        analyses: list[CointegrationAnalysis] = []

        with ProcessPoolExecutor() as executor:
            for worker_analyses in executor.map(
                CointegrationWorker.analyze,
                workers,
            ):
                analyses.extend(worker_analyses)

        return analyses
