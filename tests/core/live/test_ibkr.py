from unittest.mock import Mock, patch

from statarb.core.live.execution import LiveExecution, LiveOrder
from statarb.core.live.ibkr import IBKRClient


def test_connect() -> None:
    client = IBKRClient(
        host="localhost",
        port=7497,
        client_id=10,
    )

    client.ib = Mock()
    client.ib.isConnected.return_value = False

    client.connect()

    client.ib.connect.assert_called_once_with(
        "localhost",
        7497,
        clientId=10,
    )


def test_execute() -> None:
    client = IBKRClient()
    client.ib = Mock()

    trade = Mock()
    trade.orderStatus.status = "Filled"
    trade.order.orderId = 123

    client.ib.placeOrder.return_value = trade

    execution = LiveExecution(
        orders=[
            LiveOrder(
                ticker="7203",
                quantity=100,
                side="BUY",
            ),
        ],
    )

    with (
        patch(
            "statarb.core.live.ibkr.Stock",
        ) as stock_class,
        patch(
            "statarb.core.live.ibkr.MarketOrder",
        ) as market_order_class,
    ):
        contract = Mock()
        ib_order = Mock()

        stock_class.return_value = contract
        market_order_class.return_value = ib_order

        results = client.execute(execution)

    stock_class.assert_called_once_with(
        "7203",
        "TSEJ",
        "JPY",
    )

    client.ib.qualifyContracts.assert_called_once_with(
        contract,
    )

    market_order_class.assert_called_once_with(
        "BUY",
        100,
    )

    client.ib.placeOrder.assert_called_once_with(
        contract,
        ib_order,
    )

    assert results == [
        {
            "status": "Filled",
            "orderId": 123,
            "ticker": "7203",
            "quantity": 100,
            "side": "BUY",
        },
    ]


def test_execute_multiple_orders() -> None:
    client = IBKRClient()
    client.ib = Mock()

    trade1 = Mock()
    trade1.orderStatus.status = "Filled"
    trade1.order.orderId = 1

    trade2 = Mock()
    trade2.orderStatus.status = "Submitted"
    trade2.order.orderId = 2

    client.ib.placeOrder.side_effect = [
        trade1,
        trade2,
    ]

    execution = LiveExecution(
        orders=[
            LiveOrder(
                ticker="7203",
                quantity=100,
                side="BUY",
            ),
            LiveOrder(
                ticker="6758",
                quantity=50,
                side="SELL",
            ),
        ],
    )

    results = client.execute(execution)

    assert results == [
        {
            "status": "Filled",
            "orderId": 1,
            "ticker": "7203",
            "quantity": 100,
            "side": "BUY",
        },
        {
            "status": "Submitted",
            "orderId": 2,
            "ticker": "6758",
            "quantity": 50,
            "side": "SELL",
        },
    ]

    assert client.ib.placeOrder.call_count == 2


def test_close_disconnects_when_connected() -> None:
    client = IBKRClient()
    client.ib = Mock()
    client.ib.isConnected.return_value = True

    client.close()

    client.ib.disconnect.assert_called_once_with()


def test_close_does_nothing_when_not_connected() -> None:
    client = IBKRClient()
    client.ib = Mock()
    client.ib.isConnected.return_value = False

    client.close()

    client.ib.disconnect.assert_not_called()
