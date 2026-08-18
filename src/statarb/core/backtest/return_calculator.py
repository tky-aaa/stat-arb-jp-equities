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

        asset_returns = prices[analysis.spread.tickers].pct_change()

        beta = pd.Series(
            analysis.spread.beta.iloc[0].to_numpy(),
            index=analysis.spread.tickers,
        )

        normalized_beta = beta / beta.abs().sum()

        spread_return = asset_returns.mul(
            normalized_beta,
            axis=1,
        ).sum(axis=1)

        position = signal.position.shift(1)

        strategy_return = position * spread_return
        strategy_return = strategy_return.fillna(0.0)

        strategy_return.name = "pnl"

        return strategy_return
