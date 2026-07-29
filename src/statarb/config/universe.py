from pathlib import Path

import pandas as pd

UNIVERSE_PATH = Path(__file__).parent / "topix500.csv"


def get_topix500() -> list[str]:
    """
    Load TOPIX500 ticker universe.

    Returns
    -------
    list[str]
        TSE ticker symbols.
    """

    df = pd.read_csv(
        UNIVERSE_PATH,
    )

    return df["ticker"].astype(str).tolist()
