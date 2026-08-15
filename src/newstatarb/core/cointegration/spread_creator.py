import numpy as np
import pandas as pd

from newstatarb.config.contract import Spread


class SpreadCreator:
    def create(
        self,
        prices: pd.DataFrame,
        tickers: list[str],
        beta: np.ndarray,
        beta_index: int,
    ) -> Spread:
        beta = np.asarray(
            beta,
            dtype=float,
        ).flatten()

        if prices.shape[1] != len(beta):
            raise ValueError("Price dimension and beta dimension mismatch.")

        if len(tickers) != len(beta):
            raise ValueError("Ticker dimension and beta dimension mismatch.")

        values = prices.to_numpy() @ beta

        series = pd.Series(
            values,
            index=prices.index,
            name="spread",
        )

        return Spread(
            tickers=list(tickers),
            beta=beta.copy(),
            beta_index=beta_index,
            values=series,
        )
