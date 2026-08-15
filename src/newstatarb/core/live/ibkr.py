from dataclasses import dataclass

from ib_insync import IB, MarketOrder, Stock

from newstatarb.core.live.execution import LiveExecution


@dataclass
class IBKRClient:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1

    def __post_init__(self) -> None:
        self.ib = IB()

    def connect(self) -> None:
        if not self.ib.isConnected():
            self.ib.connect(
                self.host,
                self.port,
                clientId=self.client_id,
            )

    def execute(
        self,
        execution: LiveExecution,
    ) -> list[dict]:
        results = []

        for order in execution.orders:
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

            results.append(
                {
                    "status": trade.orderStatus.status,
                    "orderId": trade.order.orderId,
                    "ticker": order.ticker,
                    "quantity": order.quantity,
                    "side": order.side,
                }
            )

        return results

    def close(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
