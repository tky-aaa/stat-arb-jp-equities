import pandas as pd
import yfinance as yf

from newstatarb.core.data.instruments.japan_equity import JapanEquity
from newstatarb.core.data.loaders.base import PriceDataLoader


class YahooFinanceLoader(PriceDataLoader):
    def get_prices(
        self,
        instruments: list[JapanEquity],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        prices = []

        for instrument in instruments:
            data = yf.download(
                instrument.yahoo_symbol(),
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if data.empty:
                continue

            close = data["Close"]

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            close = close.rename(instrument.ticker)
            prices.append(close)

        if not prices:
            raise ValueError("No price data was retrieved.")

        return pd.concat(prices, axis=1)
