from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelectedSpread:
    """
    Tradable spread candidate.

    One object represents
    one cointegration vector.
    """

    tickers: list[str]

    beta: list[float]

    rank: int

    beta_index: int

    score: float

    half_life: float

    persistence: float


def select_top_spreads(
    ranked_results: pd.DataFrame,
    *,
    n_spreads: int = 10,
) -> list[SelectedSpread]:

    selected = []

    top = ranked_results.head(
        n_spreads,
    )

    for _, row in top.iterrows():
        beta = np.asarray(
            row["beta"],
            dtype=float,
        )

        if beta.ndim != 1:
            raise ValueError("Selected spread beta must be 1-dimensional.")

        selected.append(
            SelectedSpread(
                tickers=list(row["tickers"]),
                beta=[float(x) for x in beta],
                rank=int(row["rank"]),
                beta_index=int(row["beta_index"]),
                score=float(row["score"]),
                half_life=float(row["half_life"]),
                persistence=float(row["persistence"]),
            )
        )

    return selected
