from dataclasses import dataclass, field

from statarb.config.contract import Prices
from statarb.core.data.instruments.japan_equity import JapanEquity
from statarb.core.data.loaders.base import PriceDataLoader
from statarb.core.data.loaders.yfinance import YahooFinanceLoader
from statarb.core.data.universe.universe import Universe


@dataclass(frozen=True)
class DataAPI:
    universe: str
    training_start: str
    training_end: str
    test_start: str
    test_end: str

    loader: PriceDataLoader = field(
        default_factory=YahooFinanceLoader,
        repr=False,
    )

    def service(self) -> Prices:
        tickers = Universe(self.universe).get_tickers()
        instruments = [JapanEquity(ticker) for ticker in tickers]

        print("[DataAPI] training: loading prices...", flush=True)

        training = self.loader.get_prices(
            instruments=instruments,
            start=self.training_start,
            end=self.training_end,
        )

        valid_tickers = training.columns[training.notna().any(axis=0)].tolist()

        training = training[valid_tickers]

        print(
            f"[DataAPI] training: done {training.shape}",
            flush=True,
        )

        valid_instruments = [
            instrument
            for instrument in instruments
            if instrument.ticker in valid_tickers
        ]

        print("[DataAPI] test: loading prices...", flush=True)

        test = self.loader.get_prices(
            instruments=valid_instruments,
            start=self.test_start,
            end=self.test_end,
        )

        test = test[valid_tickers]

        print(
            f"[DataAPI] test: done {test.shape}",
            flush=True,
        )

        return Prices(
            training=training,
            test=test,
        )
