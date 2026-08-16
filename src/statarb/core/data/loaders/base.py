from abc import ABC, abstractmethod

import pandas as pd

from statarb.core.data.instruments.japan_equity import JapanEquity


class PriceDataLoader(ABC):
    @abstractmethod
    def get_prices(
        self,
        instruments: list[JapanEquity],
        start: str,
        end: str,
    ) -> pd.DataFrame: ...
