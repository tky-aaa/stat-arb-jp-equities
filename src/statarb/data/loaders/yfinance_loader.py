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

        contracts = [instrument.yahoo_contract() for instrument in instruments]

        data = yf.download(
            contracts,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

        prices = data["Close"].copy()

        prices.columns = [instrument.ticker for instrument in instruments]

        return prices
