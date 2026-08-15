from dataclasses import dataclass

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
)
from alpaca.trading.requests import MarketOrderRequest


@dataclass(frozen=True)
class AlpacaOrder:
    """
    Alpaca order representation.
    """

    ticker: str

    quantity: int

    side: str


class AlpacaExecution:
    """
    Alpaca execution interface.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        paper: bool = True,
    ):
        self.client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )

    def connect(self) -> None:
        """
        Connectivity check.
        """

        self.client.get_account()

    def submit_order(
        self,
        order: AlpacaOrder,
    ) -> dict:

        request = MarketOrderRequest(
            symbol=order.ticker,
            qty=order.quantity,
            side=(OrderSide.BUY if order.side.upper() == "BUY" else OrderSide.SELL),
            time_in_force=TimeInForce.DAY,
        )

        result = self.client.submit_order(
            order_data=request,
        )

        return {
            "status": result.status,
            "orderId": result.id,
            "ticker": order.ticker,
            "quantity": order.quantity,
            "side": order.side,
        }

    def close(self) -> None:
        """
        Nothing to close.
        """

        return
