from dataclasses import dataclass

from statarb.config.contract import (
    BacktestResult,
    PortfolioDecision,
)
from statarb.core.backtest.portfolio_return_calculator import (
    PortfolioReturnCalculator,
)
from statarb.core.backtest.return_calculator import ReturnCalculator


@dataclass(frozen=True)
class BacktestAPI:
    initial_capital: float = 1.0

    def service(
        self,
        decision: PortfolioDecision,
    ) -> BacktestResult:

        if len(decision.analyses) != len(decision.weights):
            raise ValueError("Number of analyses and weights must match.")

        if len(decision.analyses) != len(decision.signals):
            raise ValueError("Number of analyses and signals must match.")

        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")

        return_calculator = ReturnCalculator()
        portfolio_return_calculator = PortfolioReturnCalculator()

        returns = [return_calculator.calculate(signal) for signal in decision.signals]

        portfolio_pnl = portfolio_return_calculator.calculate(
            returns,
            decision.weights,
        )

        cumulative_pnl = portfolio_pnl.cumsum()

        equity = self.initial_capital + cumulative_pnl

        return BacktestResult(
            pnl=portfolio_pnl,
            cumulative_pnl=cumulative_pnl,
            equity=equity,
        )
