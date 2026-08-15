import pandas as pd


class SignalGenerator:
    def generate(
        self,
        zscore: pd.Series,
        *,
        entry_threshold: float,
        exit_threshold: float,
    ) -> pd.Series:
        if entry_threshold <= 0:
            raise ValueError("entry_threshold must be positive.")

        if exit_threshold < 0:
            raise ValueError("exit_threshold must be non-negative.")

        if exit_threshold >= entry_threshold:
            raise ValueError("exit_threshold must be smaller than entry_threshold.")

        position = 0
        positions = []

        for value in zscore:
            if pd.isna(value):
                positions.append(position)
                continue

            if position == 0 and value > entry_threshold:
                position = -1

            elif position == 0 and value < -entry_threshold:
                position = 1

            elif (
                position == 1
                and value >= -exit_threshold
                or position == -1
                and value <= exit_threshold
            ):
                position = 0

            positions.append(position)

        return pd.Series(
            positions,
            index=zscore.index,
            name="position",
        )
