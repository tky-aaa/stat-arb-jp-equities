import pandas as pd
import yfinance as yf

from statarb.core.data.instruments.japan_equity import JapanEquity
from statarb.core.data.loaders.base import PriceDataLoader


class YahooFinanceLoader(PriceDataLoader):
    def get_prices(
        self,
        instruments: list[JapanEquity],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        symbols = [instrument.yahoo_symbol() for instrument in instruments]

        data = yf.download(
            symbols,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if data.empty:
            raise ValueError("No price data was retrieved.")

        close = data["Close"]

        if isinstance(close, pd.Series):
            close = close.rename(
                instruments[0].ticker,
            ).to_frame()

        else:
            close = close.rename(
                columns={
                    instrument.yahoo_symbol(): instrument.ticker
                    for instrument in instruments
                },
            )

        return close
