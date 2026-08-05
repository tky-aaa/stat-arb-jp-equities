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

    assert result["status"] in {
        "Submitted",
        "PreSubmitted",
    }

    assert result["ticker"] == "7203"
    assert result["quantity"] == 100
    assert result["side"] == "BUY"

    broker.close()
