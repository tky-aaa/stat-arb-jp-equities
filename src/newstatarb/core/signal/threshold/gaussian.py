class GaussianThresholdOptimizer:
    """
    Optimize the threshold under the standard normal model.

    The optimal threshold is the analytical solution of

        argmax_{s > 0} s * (1 - Phi(s)).

    The solution is a known constant, so no training data
    or numerical optimization is required.
    """

    _OPTIMAL_THRESHOLD = 0.7517911919243928

    def optimize(self) -> float:
        return self._OPTIMAL_THRESHOLD
