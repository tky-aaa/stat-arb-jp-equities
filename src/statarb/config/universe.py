from pathlib import Path

import pandas as pd

UNIVERSE_PATH = Path(__file__).parent


def _load_topix_csv(
    filename: str,
) -> list[str]:
    """
    Load ticker universe.
    """

    path = UNIVERSE_PATH / filename

    df = pd.read_csv(
        path,
    )

    return df["ticker"].astype(str).tolist()


def get_topix500() -> list[str]:
    """
    TOPIX500 universe.
    """

    return _load_topix_csv(
        "topix500.csv",
    )


def get_topix100() -> list[str]:
    """
    TOPIX100 universe.
    """

    return _load_topix_csv(
        "topix100.csv",
    )
