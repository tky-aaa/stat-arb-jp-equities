import pandas as pd
import yfinance as yf

from .base import PriceDataSource


class YahooFinanceSource(PriceDataSource):
    def get_prices(
        self,
        instruments,
        start,
        end,
    ) -> pd.DataFrame:

        prices = []

        for i, instrument in enumerate(instruments):
            contract = instrument.yahoo_contract()

            data = yf.download(
                contract,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if data.empty:
                continue

            series = data["Close"]
            series.name = instrument.ticker

            prices.append(series)

        return pd.concat(
            prices,
            axis=1,
        )
