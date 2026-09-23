import pandas as pd

from statarb.config.contract import Prices
from statarb.core.data.api import DataAPI


class FakePriceDataLoader:
    def __init__(self) -> None:
        self.calls = []

    def get_prices(
        self,
        instruments,
        start,
        end,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "tickers": [instrument.ticker for instrument in instruments],
                "start": start,
                "end": end,
            }
        )

        return pd.DataFrame(
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


def test_data_api_returns_prices() -> None:
    loader = FakePriceDataLoader()

    api = DataAPI(
        universe="topix10",
        training_start="2025-01-01",
        training_end="2025-06-01",
        test_start="2025-06-01",
        test_end="2026-01-01",
        loader=loader,
    )

    prices = api.service()

    assert isinstance(prices, Prices)
    assert isinstance(prices.training, pd.DataFrame)
    assert isinstance(prices.test, pd.DataFrame)


def test_data_api_passes_requested_periods_to_loader() -> None:
    loader = FakePriceDataLoader()

    api = DataAPI(
        universe="topix10",
        training_start="2025-01-01",
        training_end="2025-06-01",
        test_start="2025-06-01",
        test_end="2026-01-01",
        loader=loader,
    )

    api.service()

    assert len(loader.calls) == 2

    assert loader.calls[0]["start"] == "2025-01-01"
    assert loader.calls[0]["end"] == "2025-06-01"

    assert loader.calls[1]["start"] == "2025-06-01"
    assert loader.calls[1]["end"] == "2026-01-01"


def test_data_api_uses_training_valid_tickers_for_test() -> None:
    loader = FakePriceDataLoader()

    api = DataAPI(
        universe="topix10",
        training_start="2025-01-01",
        training_end="2025-06-01",
        test_start="2025-06-01",
        test_end="2026-01-01",
        loader=loader,
    )

    api.service()

    assert len(loader.calls) == 2

    training_tickers = loader.calls[0]["tickers"]
    test_tickers = loader.calls[1]["tickers"]

    assert test_tickers == [
        ticker for ticker in training_tickers if ticker in ["7203", "6758"]
    ]


def test_data_api_excludes_tickers_without_training_data() -> None:
    class MissingTrainingDataLoader:
        def __init__(self) -> None:
            self.calls = []

        def get_prices(
            self,
            instruments,
            start,
            end,
        ) -> pd.DataFrame:
            self.calls.append(
                {
                    "tickers": [instrument.ticker for instrument in instruments],
                    "start": start,
                    "end": end,
                }
            )

            if start == "2025-01-01":
                return pd.DataFrame(
                    {
                        "7203": [100.0, 101.0],
                        "6758": [200.0, 202.0],
                        "285A": [float("nan"), float("nan")],
                    },
                    index=pd.date_range(
                        "2025-01-01",
                        periods=2,
                        freq="D",
                    ),
                )

            return pd.DataFrame(
                {
                    "7203": [102.0, 103.0],
                    "6758": [203.0, 204.0],
                },
                index=pd.date_range(
                    "2025-06-01",
                    periods=2,
                    freq="D",
                ),
            )

    loader = MissingTrainingDataLoader()

    api = DataAPI(
        universe="topix10",
        training_start="2025-01-01",
        training_end="2025-06-01",
        test_start="2025-06-01",
        test_end="2026-01-01",
        loader=loader,
    )

    prices = api.service()

    assert "285A" not in prices.training.columns
    assert "285A" not in prices.test.columns

    assert "285A" not in loader.calls[1]["tickers"]
