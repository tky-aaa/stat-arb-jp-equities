from pathlib import Path

import pandas as pd


def save_summary(
    summary: pd.DataFrame,
    filename: str = "summary.csv",
    output_dir: str = "results",
) -> Path:
    """
    Save one summary dataframe.
    """

    path = Path(output_dir)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = path / filename

    summary.to_csv(
        output_file,
        index=False,
    )

    return output_file


def save_selected_spreads(
    selected,
    filename: str = "selected_spreads.pkl",
    output_dir: str = "results",
) -> Path:
    """
    Save selected spreads.
    """

    path = Path(output_dir)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = path / filename

    pd.to_pickle(
        selected,
        output_file,
    )

    return output_file
