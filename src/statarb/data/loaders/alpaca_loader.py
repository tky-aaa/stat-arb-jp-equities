import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .base import PriceDataSource


class AlpacaSource(PriceDataSource):
    def __init__(
        self,
        api_key,
        secret_key,
    ):

        self.client = StockHistoricalDataClient(
            api_key,
            secret_key,
        )

    def get_prices(
        self,
        instruments,
        start,
        end,
    ) -> pd.DataFrame:

        contracts = [instrument.ticker for instrument in instruments]

        request = StockBarsRequest(
            symbol_or_symbols=contracts,
            timeframe=TimeFrame.Day,
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
        )

        bars = self.client.get_stock_bars(request)

        return bars.df["close"].unstack(level=0)
