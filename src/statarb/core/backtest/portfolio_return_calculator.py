import pandas as pd


class PortfolioReturnCalculator:
    def calculate(
        self,
        returns: list[pd.Series],
        weights: list[float],
    ) -> pd.Series:

        if len(returns) != len(weights):
            raise ValueError("Number of returns and weights must match.")

        if not returns:
            return pd.Series(
                dtype=float,
                name="portfolio_return",
            )

        weighted_returns = []

        for return_series, weight in zip(
            returns,
            weights,
        ):
            weighted_returns.append(return_series * weight)

        portfolio_returns = pd.concat(
            weighted_returns,
            axis=1,
        ).sum(
            axis=1,
            min_count=1,
        )

        portfolio_returns.name = "portfolio_return"

        return portfolio_returns
