from dataclasses import dataclass

from ib_insync import IB, MarketOrder, Stock


@dataclass(frozen=True)
class IBKROrder:
    ticker: str
    quantity: int
    side: str


class IBKRExecution:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id

        self.ib = IB()

    def connect(self) -> None:
        if not self.ib.isConnected():
            self.ib.connect(
                self.host,
                self.port,
                clientId=self.client_id,
            )

    def submit_order(
        self,
        order: IBKROrder,
    ) -> dict:

        contract = Stock(
            order.ticker,
            "TSEJ",
            "JPY",
        )

        self.ib.qualifyContracts(contract)

        ib_order = MarketOrder(
            order.side,
            order.quantity,
        )

        trade = self.ib.placeOrder(
            contract,
            ib_order,
        )

        self.ib.sleep(1)

        return {
            "status": trade.orderStatus.status,
            "orderId": trade.order.orderId,
            "ticker": order.ticker,
            "quantity": order.quantity,
            "side": order.side,
        }

    def close(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
