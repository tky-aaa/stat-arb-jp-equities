from newstatarb.core.data.universe.universe import Universe


def test_topix10_returns_tickers() -> None:
    tickers = Universe("topix10").get_tickers()

    assert isinstance(tickers, list)
    assert len(tickers) > 0
    assert all(isinstance(ticker, str) for ticker in tickers)


def test_unknown_universe_is_rejected() -> None:
    try:
        Universe("unknown").get_tickers()
    except ValueError as exc:
        assert str(exc) == "Unknown universe: unknown"
    else:
        raise AssertionError("ValueError was not raised")
