import numpy as np
import pandas as pd

from statarb.config.contract import Spread


class SpreadCreator:
    def create(
        self,
        prices: pd.DataFrame,
        tickers: list[str],
        beta: np.ndarray,
    ) -> Spread:
        beta = np.asarray(
            beta,
            dtype=float,
        ).flatten()

        if prices.shape[1] != len(beta):
            raise ValueError(
                "Price dimension and beta dimension mismatch.",
            )

        if len(tickers) != len(beta):
            raise ValueError(
                "Ticker dimension and beta dimension mismatch.",
            )

        beta_frame = pd.DataFrame(
            np.tile(
                beta,
                (len(prices), 1),
            ),
            index=prices.index,
            columns=tickers,
        )

        intercept = pd.Series(
            0.0,
            index=prices.index,
            name="intercept",
        )

        values = prices.to_numpy() @ beta

        series = pd.Series(
            values,
            index=prices.index,
            name="spread",
        )

        return Spread(
            tickers=list(tickers),
            beta=beta_frame,
            intercept=intercept,
            values=series,
        )
