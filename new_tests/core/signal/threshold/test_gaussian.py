from newstatarb.core.signal.threshold.gaussian import (
    GaussianThresholdOptimizer,
)


def test_optimize_returns_analytical_optimum() -> None:
    optimizer = GaussianThresholdOptimizer()

    threshold = optimizer.optimize()

    assert threshold == 0.7517911919243928
