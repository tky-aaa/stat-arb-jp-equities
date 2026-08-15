import numpy as np

from newstatarb.config.contract import (
    BacktestMetrics,
    BacktestResult,
    Signal,
)


class BacktestMetricsCalculator:
    def calculate(
        self,
        result: BacktestResult,
        signals: list[Signal],
        *,
        periods_per_year: int = 252,
    ) -> BacktestMetrics:

        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive.")

        if result.equity.empty:
            raise ValueError("Backtest result must not be empty.")

        if result.pnl.empty:
            raise ValueError("Backtest result must not be empty.")

        initial_equity = result.equity.iloc[0]

        if initial_equity == 0:
            raise ValueError("Initial equity must not be zero.")

        total_return = float(
            result.equity.iloc[-1] / initial_equity - 1.0,
        )

        periods = len(result.equity)

        annualized_return = float(
            (1.0 + total_return) ** (periods_per_year / periods) - 1.0,
        )

        volatility = float(
            result.pnl.std() * np.sqrt(periods_per_year),
        )

        if volatility == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = float(
                result.pnl.mean() * periods_per_year / volatility,
            )

        running_max = result.equity.cummax()

        drawdown = (result.equity - running_max) / running_max

        max_drawdown = float(
            drawdown.min(),
        )

        non_zero_pnl = result.pnl[result.pnl != 0]

        if len(non_zero_pnl) == 0:
            win_rate = 0.0
        else:
            win_rate = float(
                (non_zero_pnl > 0).mean(),
            )

        number_of_trades = 0

        for signal in signals:
            number_of_trades += int((signal.position.diff().fillna(0).abs() > 0).sum())

        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            number_of_trades=number_of_trades,
        )
