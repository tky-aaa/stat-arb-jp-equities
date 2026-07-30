from statarb.execution.alpaca import (
    AlpacaExecution,
    AlpacaOrder,
)


def test_alpaca_execution_interface():

    broker = AlpacaExecution(
        paper=True,
    )

    broker.connect()

    result = broker.submit_order(
        AlpacaOrder(
            ticker="AAPL",
            quantity=10,
            side="BUY",
        )
    )

    assert result["status"] == "submitted"
    assert result["ticker"] == "AAPL"
    assert result["paper"] is True

    broker.close()
