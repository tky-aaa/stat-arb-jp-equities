import pandas as pd

from newstatarb.core.data.instruments.japan_equity import JapanEquity
from newstatarb.core.data.loaders.yfinance import YahooFinanceLoader


def test_yfinance_loader_returns_close_prices(
    monkeypatch,
) -> None:
    def fake_download(
        symbol,
        start,
        end,
        auto_adjust,
        progress,
        threads,
    ):
        assert symbol == "7203.T"
        assert start == "2025-01-01"
        assert end == "2025-02-01"

        return pd.DataFrame(
            {
                "Close": [100.0, 101.0],
            },
            index=pd.date_range(
                "2025-01-01",
                periods=2,
                freq="D",
            ),
        )

    monkeypatch.setattr(
        "newstatarb.core.data.loaders.yfinance.yf.download",
        fake_download,
    )

    loader = YahooFinanceLoader()

    prices = loader.get_prices(
        instruments=[JapanEquity("7203")],
        start="2025-01-01",
        end="2025-02-01",
    )

    expected = pd.DataFrame(
        {
            "7203": [100.0, 101.0],
        },
        index=pd.date_range(
            "2025-01-01",
            periods=2,
            freq="D",
        ),
    )

    pd.testing.assert_frame_equal(
        prices,
        expected,
    )


def test_yfinance_loader_raises_when_no_data(
    monkeypatch,
) -> None:
    def fake_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(
        "newstatarb.core.data.loaders.yfinance.yf.download",
        fake_download,
    )

    loader = YahooFinanceLoader()

    try:
        loader.get_prices(
            instruments=[JapanEquity("7203")],
            start="2025-01-01",
            end="2025-02-01",
        )
    except ValueError as exc:
        assert str(exc) == "No price data was retrieved."
    else:
        raise AssertionError("ValueError was not raised")
