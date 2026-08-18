from dataclasses import dataclass

import numpy as np
import pandas as pd

from statarb.config.contract import (
    CointegrationAnalysis,
    Prices,
    Signal,
    Spread,
)
from statarb.core.cointegration.spread_creator import SpreadCreator
from statarb.core.signal.generator import SignalGenerator
from statarb.core.signal.kalman import KalmanFilter
from statarb.core.signal.threshold.empirical import (
    EmpiricalThresholdOptimizer,
)
from statarb.core.signal.threshold.fixed import FixedThreshold
from statarb.core.signal.threshold.gaussian import (
    GaussianThresholdOptimizer,
)
from statarb.core.signal.zscore import ZScoreCalculator


@dataclass(frozen=True)
class SignalAPI:
    zscore_window: int
    entry_threshold: float
    exit_threshold: float
    spread_method: str
    threshold_method: str = "fixed"
    min_threshold: float = 0.5
    max_threshold: float = 3.0
    n_thresholds: int = 100
    smoothing_lambda: float = 10.0
    kalman_alpha: float = 1e-5

    def service(
        self,
        prices: Prices,
        analyses: list[CointegrationAnalysis],
    ) -> list[Signal]:
        zscore_calculator = ZScoreCalculator()
        signal_generator = SignalGenerator()

        signals: list[Signal] = []

        for analysis in analyses:
            training_spread, test_spread = self._create_spreads(
                prices=prices,
                analysis=analysis,
            )

            if self.threshold_method == "empirical":
                training_zscore = zscore_calculator.calculate(
                    training_spread.values,
                    window=self.zscore_window,
                )

                entry_threshold = self._determine_entry_threshold(
                    training_zscore=training_zscore,
                )
            else:
                entry_threshold = self._determine_entry_threshold(
                    training_zscore=None,
                )

            test_zscore = zscore_calculator.calculate(
                test_spread.values,
                window=self.zscore_window,
            )

            position = signal_generator.generate(
                test_zscore,
                entry_threshold=entry_threshold,
                exit_threshold=self.exit_threshold,
            )

            signals.append(
                Signal(
                    spread=test_spread.values,
                    zscore=test_zscore,
                    position=position,
                    beta=test_spread.beta,
                )
            )

        return signals

    def _determine_entry_threshold(
        self,
        *,
        training_zscore: pd.Series | None,
    ) -> float:
        if self.threshold_method == "fixed":
            return FixedThreshold(
                self.entry_threshold,
            ).optimize()

        if self.threshold_method == "gaussian":
            return GaussianThresholdOptimizer().optimize()

        if self.threshold_method == "empirical":
            if training_zscore is None:
                raise ValueError(
                    "training_zscore is required for empirical threshold.",
                )

            return EmpiricalThresholdOptimizer(
                min_threshold=self.min_threshold,
                max_threshold=self.max_threshold,
                n_thresholds=self.n_thresholds,
                smoothing_lambda=self.smoothing_lambda,
            ).optimize(
                training_zscore,
            )

        raise ValueError(
            f"Unknown threshold_method: {self.threshold_method}",
        )

    def _create_spreads(
        self,
        *,
        prices: Prices,
        analysis: CointegrationAnalysis,
    ) -> tuple[Spread, Spread]:
        tickers = analysis.spread.tickers

        training_prices = prices.training[tickers]
        test_prices = prices.test[tickers]

        if self.spread_method == "fixed":
            spread_creator = SpreadCreator()

            beta = np.asarray(
                analysis.spread.beta,
                dtype=float,
            )

            training_spread = spread_creator.create(
                prices=training_prices,
                tickers=tickers,
                beta=beta,
            )

            test_spread = spread_creator.create(
                prices=test_prices,
                tickers=tickers,
                beta=beta,
            )

            return training_spread, test_spread

        if self.spread_method == "kalman":
            kalman_filter = KalmanFilter()

            (
                initial_state,
                initial_covariance,
                observation_variance,
                transition_covariance,
            ) = kalman_filter.initialize(
                training_prices,
                alpha=self.kalman_alpha,
            )

            (
                training_values,
                training_betas,
                training_intercept,
                state,
                covariance,
            ) = kalman_filter.filter(
                training_prices,
                state=initial_state,
                covariance=initial_covariance,
                observation_variance=observation_variance,
                transition_covariance=transition_covariance,
            )

            (
                test_values,
                test_betas,
                test_intercept,
                _,
                _,
            ) = kalman_filter.filter(
                test_prices,
                state=state,
                covariance=covariance,
                observation_variance=observation_variance,
                transition_covariance=transition_covariance,
            )

            training_spread = Spread(
                tickers=tickers,
                beta=training_betas,
                intercept=training_intercept,
                values=training_values,
            )

            test_spread = Spread(
                tickers=tickers,
                beta=test_betas,
                intercept=test_intercept,
                values=test_values,
            )

            return training_spread, test_spread

        raise ValueError(
            f"Unknown spread_method: {self.spread_method}",
        )
