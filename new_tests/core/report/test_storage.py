from pathlib import Path

import pandas as pd

from newstatarb.core.report.storage import ReportStorage


def test_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "result.pkl"

    output = pd.Series(
        [1.0, 2.0, 3.0],
        name="result",
    )

    storage = ReportStorage()

    storage.save(
        output,
        path,
    )

    loaded = storage.load(path)

    pd.testing.assert_series_equal(
        loaded,
        output,
    )
