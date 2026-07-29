import pandas as pd
from ib_insync import IB, util

from .base import PriceDataSource


class IBKRSource(PriceDataSource):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 1,
    ):

        self.ib = IB()

        self.ib.connect(
            host,
            port,
            clientId=client_id,
        )

        # delayed market data
        self.ib.reqMarketDataType(4)

    def get_prices(
        self,
        instruments,
        start: str,
        end: str,
    ) -> pd.DataFrame:

        prices = {}

        end_datetime = pd.Timestamp(end).strftime("%Y%m%d %H:%M:%S UTC")

        for instrument in instruments:
            contract = instrument.ibkr_contract()

            self.ib.qualifyContracts(contract)

            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=end_datetime,
                durationStr="1 Y",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )

            if not bars:
                raise RuntimeError(f"No historical data returned: {instrument.ticker}")

            df = util.df(bars)

            df["date"] = pd.to_datetime(df["date"])

            df = df.set_index("date")

            prices[instrument.ticker] = df["close"]

        return pd.DataFrame(prices)
