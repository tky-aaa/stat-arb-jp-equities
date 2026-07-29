from abc import ABC, abstractmethod

import pandas as pd

from ..instruments.japan_equity import JapanEquity


class PriceDataSource(ABC):
    @abstractmethod
    def get_prices(
        self,
        instruments: list[JapanEquity],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        pass
