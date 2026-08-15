import numpy as np
import pandas as pd

from newstatarb.config.contract import Spread
from newstatarb.core.cointegration.stationarity_evaluator import (
    StationarityEvaluator,
)


def test_evaluate_returns_adf_and_kpss_statistics() -> None:
    rng = np.random.default_rng(42)

    values = []
    x = 0.0

    for _ in range(200):
        x = 0.5 * x + rng.normal(0.0, 1.0)
        values.append(x)

    spread = Spread(
        tickers=["AAA", "BBB"],
        beta=np.array([1.0, -1.0]),
        beta_index=0,
        values=pd.Series(values, name="spread"),
    )

    evaluator = StationarityEvaluator()

    result = evaluator.evaluate(spread)

    assert set(result) == {
        "adf_stat",
        "adf_pvalue",
        "kpss_stat",
        "kpss_pvalue",
    }

    assert np.isfinite(result["adf_stat"])
    assert np.isfinite(result["adf_pvalue"])
    assert np.isfinite(result["kpss_stat"])
    assert np.isfinite(result["kpss_pvalue"])

    assert result["adf_pvalue"] < 0.05
    assert result["kpss_pvalue"] > 0.05
