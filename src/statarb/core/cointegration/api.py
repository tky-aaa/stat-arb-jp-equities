from dataclasses import dataclass

import numpy as np

from statarb.config.contract import (
    CointegrationAnalysis,
    Prices,
    SpreadEvaluation,
    SubGroup,
)
from statarb.core.cointegration.distribution_evaluator import (
    DistributionEvaluator,
)
from statarb.core.cointegration.johansen_tester import (
    JohansenTester,
)
from statarb.core.cointegration.mean_reversion_evaluator import (
    MeanReversionEvaluator,
)
from statarb.core.cointegration.spread_creator import (
    SpreadCreator,
)
from statarb.core.cointegration.stationarity_evaluator import (
    StationarityEvaluator,
)


@dataclass(frozen=True)
class CointegrationAPI:
    persistence_days: int

    def service(
        self,
        prices: Prices,
        subgroups: list[SubGroup],
    ) -> list[CointegrationAnalysis]:

        log_prices = np.log(prices.training)

        johansen_tester = JohansenTester()
        spread_creator = SpreadCreator()
        stationarity_evaluator = StationarityEvaluator()
        mean_reversion_evaluator = MeanReversionEvaluator()
        distribution_evaluator = DistributionEvaluator()

        analyses = []

        total = len(subgroups)

        for i, subgroup in enumerate(subgroups, start=1):
            selected_log_prices = log_prices[subgroup.tickers]

            result = johansen_tester.estimate_cointegration(
                log_prices=selected_log_prices,
            )

            if result.rank <= 0:
                self._print_progress(i, total)
                continue

            for beta_index in range(result.rank):
                beta = result.beta[:, beta_index]

                spread = spread_creator.create(
                    prices=selected_log_prices,
                    tickers=result.tickers,
                    beta=beta,
                    beta_index=beta_index,
                )

                stationarity = stationarity_evaluator.evaluate(
                    spread,
                )

                mean_reversion = mean_reversion_evaluator.evaluate(
                    spread,
                    persistence_window=self.persistence_days,
                )

                distribution = distribution_evaluator.evaluate(
                    spread,
                )

                evaluation = SpreadEvaluation(
                    **stationarity,
                    **mean_reversion,
                    **distribution,
                )

                analyses.append(
                    CointegrationAnalysis(
                        tickers=result.tickers,
                        rank=result.rank,
                        beta_index=beta_index,
                        beta=beta,
                        spread=spread,
                        evaluation=evaluation,
                    )
                )

            self._print_progress(i, total)

        print()

        return analyses

    @staticmethod
    def _print_progress(
        current: int,
        total: int,
        width: int = 30,
    ) -> None:
        if total == 0:
            return

        ratio = current / total
        filled = int(width * ratio)

        bar = "█" * filled + "░" * (width - filled)

        print(
            f"\r[3/6] CointegrationAPI: [{bar}] {ratio:6.2%} ({current}/{total})",
            end="",
            flush=True,
        )
