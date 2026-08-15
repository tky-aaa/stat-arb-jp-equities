import pandas as pd


class ZScoreCalculator:
    def calculate(
        self,
        spread: pd.Series,
        *,
        window: int,
    ) -> pd.Series:
        if window <= 1:
            raise ValueError("window must be greater than 1.")

        rolling_mean = spread.rolling(
            window,
        ).mean()

        rolling_std = spread.rolling(
            window,
        ).std()

        zscore = (spread - rolling_mean) / rolling_std

        zscore.name = "zscore"

        return zscore
