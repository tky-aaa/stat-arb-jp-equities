import numpy as np
import pandas as pd

from newstatarb.config.contract import Spread
from newstatarb.core.cointegration.distribution_evaluator import (
    DistributionEvaluator,
)


def test_evaluate_returns_distribution_statistics() -> None:
    values = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        name="spread",
    )

    spread = Spread(
        tickers=["AAA", "BBB"],
        beta=np.array([1.0, -1.0]),
        beta_index=0,
        values=values,
    )

    evaluator = DistributionEvaluator()

    result = evaluator.evaluate(spread)

    assert result["mean"] == 2.5
    assert np.isclose(
        result["variance"],
        values.var(),
    )
    assert np.isclose(
        result["std"],
        values.std(),
    )


def test_evaluate_rejects_too_few_observations() -> None:
    values = pd.Series(
        [1.0],
        name="spread",
    )

    spread = Spread(
        tickers=["AAA", "BBB"],
        beta=np.array([1.0, -1.0]),
        beta_index=0,
        values=values,
    )

    evaluator = DistributionEvaluator()

    try:
        evaluator.evaluate(spread)
    except ValueError:
        return

    raise AssertionError("Expected ValueError.")
