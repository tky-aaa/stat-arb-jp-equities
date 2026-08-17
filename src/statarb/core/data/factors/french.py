import io
import zipfile

import pandas as pd
import requests

from .base import FactorDataSource


class FrenchFactorSource(FactorDataSource):
    """
    Ken French Data Library factor source.

    Provides Fama-French 3 factors.
    """

    URL = (
        "https://mba.tuck.dartmouth.edu/"
        "pages/faculty/ken.french/"
        "ftp/Japan_3_Factors_Daily_CSV.zip"
    )
    # joint

    def get_factors(
        self,
        start: str,
        end: str,
    ) -> pd.DataFrame:

        response = requests.get(
            self.URL,
            timeout=30,
        )

        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            filename = z.namelist()[0]

            with z.open(filename) as f:
                raw = pd.read_csv(
                    f,
                    skiprows=9,
                )

        raw = raw[raw.iloc[:, 0].astype(str).str.match(r"\d{8}")]

        raw.columns = [
            "date",
            "MKT",
            "SMB",
            "HML",
            "RF",
        ]

        raw["date"] = pd.to_datetime(
            raw["date"],
            format="%Y%m%d",
        )

        raw = raw.set_index("date")

        raw = raw.astype(float)

        # percentage → decimal
        raw = raw / 100

        return raw.loc[start:end]
