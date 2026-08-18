import pandas as pd

from statarb.config.contract import (
    CointegrationAnalysis,
    Signal,
)


class ReturnCalculator:
    def calculate(
        self,
        prices: pd.DataFrame,
        analysis: CointegrationAnalysis,
        signal: Signal,
    ) -> pd.Series:
        tickers = analysis.spread.tickers

        asset_returns = prices[tickers].pct_change()

        beta = signal.beta[tickers]
        normalized_beta = beta.div(
            beta.abs().sum(axis=1),
            axis=0,
        )

        spread_return = asset_returns.mul(
            normalized_beta,
            axis=1,
        ).sum(axis=1)

        position = signal.position.shift(1)

        strategy_return = position * spread_return
        strategy_return = strategy_return.fillna(0.0)
        strategy_return.name = "pnl"

        return strategy_return
