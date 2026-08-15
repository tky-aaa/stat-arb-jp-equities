from statarb.execution.ibkr import IBKRExecution


def test_ibkr_execution_interface():
    broker = IBKRExecution()

    assert broker.host == "127.0.0.1"
    assert broker.port == 7497
    assert broker.client_id == 1

    assert broker.ib is not None
    assert broker.ib.isConnected() is False
