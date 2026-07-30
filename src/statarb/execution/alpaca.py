from dataclasses import dataclass


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
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper

    def connect(self) -> None:
        """
        Connect to Alpaca.

        Actual API connection will be implemented later.
        """

        return

    def submit_order(
        self,
        order: AlpacaOrder,
    ) -> dict:
        """
        Submit order.

        Paper implementation only.
        """

        return {
            "status": "submitted",
            "ticker": order.ticker,
            "quantity": order.quantity,
            "side": order.side,
            "paper": self.paper,
        }

    def close(self) -> None:
        """
        Close connection.
        """

        return
