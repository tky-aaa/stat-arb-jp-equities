import pandas as pd

from statarb.core.data.instruments.japan_equity import JapanEquity
from statarb.core.data.loaders.yfinance import YahooFinanceLoader


def test_yfinance_loader_returns_close_prices(
    monkeypatch,
) -> None:
    def fake_download(
        symbols,
        start,
        end,
        auto_adjust,
        progress,
        threads,
    ):
        assert symbols == ["7203.T"]
        assert start == "2025-01-01"
        assert end == "2025-02-01"

        return pd.DataFrame(
            {
                ("Close", "7203.T"): [100.0, 101.0],
            },
            index=pd.date_range(
                "2025-01-01",
                periods=2,
                freq="D",
            ),
        )

    monkeypatch.setattr(
        "statarb.core.data.loaders.yfinance.yf.download",
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
        "statarb.core.data.loaders.yfinance.yf.download",
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


def test_yfinance_loader_downloads_multiple_instruments_in_one_request(
    monkeypatch,
) -> None:
    calls = []

    def fake_download(
        symbols,
        start,
        end,
        auto_adjust,
        progress,
        threads,
    ):
        calls.append(
            {
                "symbols": symbols,
                "start": start,
                "end": end,
                "auto_adjust": auto_adjust,
                "progress": progress,
                "threads": threads,
            }
        )

        return pd.DataFrame(
            {
                ("Close", "7203.T"): [100.0, 101.0],
                ("Close", "6758.T"): [200.0, 202.0],
            },
            index=pd.date_range(
                "2025-01-01",
                periods=2,
                freq="D",
            ),
        )

    monkeypatch.setattr(
        "statarb.core.data.loaders.yfinance.yf.download",
        fake_download,
    )

    loader = YahooFinanceLoader()

    prices = loader.get_prices(
        instruments=[
            JapanEquity("7203"),
            JapanEquity("6758"),
        ],
        start="2025-01-01",
        end="2025-02-01",
    )

    assert len(calls) == 1

    assert calls[0]["symbols"] == [
        "7203.T",
        "6758.T",
    ]

    assert calls[0]["start"] == "2025-01-01"
    assert calls[0]["end"] == "2025-02-01"
    assert calls[0]["auto_adjust"] is True
    assert calls[0]["progress"] is False
    assert calls[0]["threads"] is False

    expected = pd.DataFrame(
        {
            "7203": [100.0, 101.0],
            "6758": [200.0, 202.0],
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
