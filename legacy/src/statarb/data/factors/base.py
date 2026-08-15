from abc import ABC, abstractmethod

import pandas as pd


class FactorDataSource(ABC):
    """
    Abstract interface for factor data providers.
    """

    @abstractmethod
    def get_factors(
        self,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Return factor returns.

        Returns
        -------
        pd.DataFrame

        columns:
            MKT
            SMB
            HML
            RF
        """
