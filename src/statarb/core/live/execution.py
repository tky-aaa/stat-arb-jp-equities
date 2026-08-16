from dataclasses import dataclass

from statarb.config.contract import PortfolioDecision


@dataclass(frozen=True)
class LiveOrder:
    ticker: str
    quantity: int
    side: str


@dataclass(frozen=True)
class LiveExecution:
    orders: list[LiveOrder]


class LiveExecutionBuilder:
    def __init__(self, base_shares: int = 100):
        self.base_shares = base_shares

    def build(
        self,
        decision: PortfolioDecision,
    ) -> LiveExecution:
        orders = []

        for analysis, signal in zip(
            decision.analyses,
            decision.signals,
        ):
            position = signal.position.iloc[-1]

            if position == 0:
                continue

            for ticker, beta in zip(
                analysis.tickers,
                analysis.beta,
            ):
                quantity = max(
                    1,
                    round(abs(beta) * self.base_shares),
                )

                beta_side = "BUY" if beta > 0 else "SELL"

                if position < 0:
                    beta_side = "SELL" if beta_side == "BUY" else "BUY"

                orders.append(
                    LiveOrder(
                        ticker=ticker,
                        quantity=quantity,
                        side=beta_side,
                    )
                )

        return LiveExecution(
            orders=orders,
        )
