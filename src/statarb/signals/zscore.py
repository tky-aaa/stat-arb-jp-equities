from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ZScoreSignal:
    """
    Z-score based mean reversion signal.
    """

    zscore: pd.Series

    position: pd.Series


def calculate_zscore(
    spread: pd.Series,
    *,
    window: int = 60,
) -> pd.Series:
    """
    Calculate rolling z-score.

    z_t =
        spread_t - mean_t
        ---------------
             std_t
    """

    rolling_mean = spread.rolling(window).mean()

    rolling_std = spread.rolling(window).std()

    z = (spread - rolling_mean) / rolling_std

    z.name = "zscore"

    return z


def generate_zscore_signal(
    spread: pd.Series,
    *,
    window: int = 60,
    threshold: float = 2.0,
    exit_threshold: float = 0.5,
) -> ZScoreSignal:
    """
    Generate mean reversion trading signal.

    Position:

        +1:
            Long spread

        -1:
            Short spread

         0:
            Flat
    """

    z = calculate_zscore(
        spread,
        window=window,
    )

    position = pd.Series(
        0,
        index=spread.index,
        dtype=float,
    )

    # Entry

    position[z < -threshold] = 1

    position[z > threshold] = -1

    # Exit zone

    position[z.abs() < exit_threshold] = 0

    # Hold previous position

    position = (
        position.replace(
            0,
            pd.NA,
        )
        .ffill()
        .fillna(0)
    )

    position.name = "position"

    return ZScoreSignal(
        zscore=z,
        position=position,
    )
