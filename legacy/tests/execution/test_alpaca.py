import os

import pytest
from dotenv import load_dotenv

from statarb.execution.alpaca import (
    AlpacaExecution,
    AlpacaOrder,
)

load_dotenv()


@pytest.mark.skipif(
    not os.getenv("ALPACA_API_KEY"),
    reason="No Alpaca credentials.",
)
def test_alpaca_execution():

    broker = AlpacaExecution(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        paper=True,
    )

    broker.connect()

    result = broker.submit_order(
        AlpacaOrder(
            ticker="SPY",
            quantity=1,
            side="BUY",
        )
    )

    assert result["ticker"] == "SPY"
    assert result["quantity"] == 1
    assert result["side"] == "BUY"

    broker.close()
