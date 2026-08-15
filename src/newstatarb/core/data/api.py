from dataclasses import dataclass, field

from newstatarb.config.contract import Prices
from newstatarb.core.data.instruments.japan_equity import JapanEquity
from newstatarb.core.data.loaders.base import PriceDataLoader
from newstatarb.core.data.loaders.yfinance import YahooFinanceLoader
from newstatarb.core.data.universe.universe import Universe


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

        training = self.loader.get_prices(
            instruments=instruments,
            start=self.training_start,
            end=self.training_end,
        )

        test = self.loader.get_prices(
            instruments=instruments,
            start=self.test_start,
            end=self.test_end,
        )

        return Prices(
            training=training,
            test=test,
        )
