import numpy as np
import pandas as pd


class EmpiricalThresholdOptimizer:
    def __init__(
        self,
        *,
        min_threshold: float,
        max_threshold: float,
        n_thresholds: int,
        smoothing_lambda: float,
    ) -> None:
        if min_threshold <= 0:
            raise ValueError("min_threshold must be positive.")

        if max_threshold <= min_threshold:
            raise ValueError("max_threshold must be greater than min_threshold.")

        if n_thresholds < 2:
            raise ValueError("n_thresholds must be at least 2.")

        if smoothing_lambda < 0:
            raise ValueError("smoothing_lambda must be non-negative.")

        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.n_thresholds = n_thresholds
        self.smoothing_lambda = smoothing_lambda

    def optimize(
        self,
        zscore: pd.Series,
    ) -> float:
        zscore = zscore.dropna()

        if zscore.empty:
            raise ValueError("No valid z-score observations.")

        thresholds = np.linspace(
            self.min_threshold,
            self.max_threshold,
            self.n_thresholds,
        )

        empirical_frequency = np.array(
            [np.mean(zscore.to_numpy() > threshold) for threshold in thresholds],
            dtype=float,
        )

        difference = np.zeros(
            (
                self.n_thresholds - 1,
                self.n_thresholds,
            ),
            dtype=float,
        )

        for index in range(self.n_thresholds - 1):
            difference[index, index] = 1.0
            difference[index, index + 1] = -1.0

        system = (
            np.eye(self.n_thresholds)
            + self.smoothing_lambda * difference.T @ difference
        )

        smoothed_frequency = np.linalg.solve(
            system,
            empirical_frequency,
        )

        profit = thresholds * smoothed_frequency

        optimal_index = int(np.argmax(profit))

        return float(thresholds[optimal_index])
