from statarb.execution.ibkr import (
    IBKRExecution,
    IBKROrder,
)


def test_ibkr_execution_interface():

    broker = IBKRExecution()

    broker.connect()

    result = broker.submit_order(
        IBKROrder(
            ticker="7203",
            quantity=100,
            side="BUY",
        )
    )

    assert result["status"] == "submitted"
    assert result["ticker"] == "7203"

    broker.close()
