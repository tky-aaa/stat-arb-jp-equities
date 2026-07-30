from dataclasses import dataclass


@dataclass(frozen=True)
class IBKROrder:
    """
    IBKR order representation.
    """

    ticker: str

    quantity: int

    side: str


class IBKRExecution:
    """
    IBKR execution interface.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id

    def connect(self) -> None:
        """
        Connect to IBKR.

        Actual TWS connection will be implemented later.
        """

        return

    def submit_order(
        self,
        order: IBKROrder,
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
        }

    def close(self) -> None:
        """
        Close connection.
        """

        return
