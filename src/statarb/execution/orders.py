from dataclasses import dataclass

from statarb.cointegration.selection import SelectedSpread

from .ibkr import IBKROrder


@dataclass(frozen=True)
class ExecutionOrder:
    """
    Executable order set for one spread.
    """

    orders: list[IBKROrder]


def create_orders(
    spread: SelectedSpread,
    *,
    base_shares: int = 100,
) -> ExecutionOrder:
    """
    Convert one spread into executable orders.

    Parameters
    ----------
    spread:
        Selected spread.

    base_shares:
        Position size for β = 1.

    Returns
    -------
    ExecutionOrder
    """

    orders = []

    for ticker, beta in zip(
        spread.tickers,
        spread.beta,
    ):
        quantity = max(
            1,
            round(abs(beta) * base_shares),
        )

        side = "BUY" if beta > 0 else "SELL"

        orders.append(
            IBKROrder(
                ticker=ticker,
                quantity=quantity,
                side=side,
            )
        )

    return ExecutionOrder(
        orders=orders,
    )
