import numpy as np
import pandas as pd

from statarb.config.contract import Spread
from statarb.core.cointegration.mean_reversion_evaluator import (
    MeanReversionEvaluator,
)


def test_evaluate_returns_mean_reversion_metrics() -> None:
    rng = np.random.default_rng(42)

    values = []
    x = 0.0

    for _ in range(200):
        x = 0.7 * x + rng.normal(0.0, 1.0)
        values.append(x)

    values = pd.Series(values, name="spread")
    spread = Spread(
        tickers=["AAA", "BBB"],
        beta=pd.DataFrame(
            [[1.0, -1.0]] * len(values),
            index=values.index,
            columns=["AAA", "BBB"],
        ),
        intercept=pd.Series(
            0.0,
            index=values.index,
            name="intercept",
        ),
        values=values,
    )

    evaluator = MeanReversionEvaluator()

    result = evaluator.evaluate(
        spread,
        persistence_window=50,
    )

    assert set(result) == {
        "rho1",
        "phi",
        "half_life",
        "portmanteau",
        "persistence",
    }

    assert np.isfinite(result["rho1"])
    assert np.isfinite(result["phi"])
    assert np.isfinite(result["half_life"])
    assert np.isfinite(result["portmanteau"])
    assert np.isfinite(result["persistence"])
    assert 0.0 <= result["persistence"] <= 1.0

    assert 0 < result["phi"] < 1
    assert result["half_life"] > 0
