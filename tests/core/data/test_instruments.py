from statarb.core.data.instruments.japan_equity import JapanEquity


def test_japan_equity_returns_yahoo_symbol() -> None:
    instrument = JapanEquity("7203")

    assert instrument.yahoo_symbol() == "7203.T"
