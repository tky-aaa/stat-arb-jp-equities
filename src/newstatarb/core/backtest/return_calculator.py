import pandas as pd

from newstatarb.config.contract import Signal


class ReturnCalculator:
    def calculate(
        self,
        signal: Signal,
    ) -> pd.Series:

        spread_change = signal.spread.diff()

        position = signal.position.shift(1)

        strategy_pnl = position * spread_change

        strategy_pnl.name = "pnl"

        return strategy_pnl
