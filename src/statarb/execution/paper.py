from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PaperTrade:
    """
    One paper trade.
    """

    timestamp: pd.Timestamp

    position: int

    spread: float

    action: str


def simulate_paper_execution(
    spread: pd.Series,
    signal: pd.Series,
) -> pd.DataFrame:
    """
    Simulate paper execution from trading signals.
    """

    if not spread.index.equals(signal.index):
        raise ValueError("spread and signal index must match.")

    trades: list[PaperTrade] = []

    previous_position = 0

    for timestamp, position, value in zip(
        spread.index,
        signal,
        spread,
    ):
        if position == previous_position:
            continue

        if position == 1:
            action = "BUY_SPREAD"
        elif position == -1:
            action = "SELL_SPREAD"
        elif previous_position == 1:
            action = "EXIT_LONG"
        elif previous_position == -1:
            action = "EXIT_SHORT"
        else:
            action = "FLAT"

        trades.append(
            PaperTrade(
                timestamp=timestamp,
                position=int(position),
                spread=float(value),
                action=action,
            )
        )

        previous_position = int(position)

    return pd.DataFrame([trade.__dict__ for trade in trades])
