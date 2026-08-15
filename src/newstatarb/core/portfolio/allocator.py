import numpy as np

from newstatarb.config.contract import CointegrationAnalysis


class PortfolioAllocator:
    def allocate(
        self,
        analyses: list[CointegrationAnalysis],
        *,
        max_weight: float = 0.2,
    ) -> list[float]:

        if not analyses:
            return []

        if max_weight <= 0:
            raise ValueError("max_weight must be positive.")

        volatilities = []

        for analysis in analyses:
            values = analysis.spread.values.dropna().astype(float)

            if len(values) == 0:
                raise ValueError("Spread contains no valid observations.")

            volatility = float(values.std())

            if not np.isfinite(volatility) or volatility <= 0:
                raise ValueError("Spread volatility must be positive and finite.")

            volatilities.append(volatility)

        inverse_volatility = 1 / np.asarray(
            volatilities,
            dtype=float,
        )

        weights = inverse_volatility / inverse_volatility.sum()

        weights = self._cap_weights(
            weights,
            max_weight=max_weight,
        )

        return weights.tolist()

    def _cap_weights(
        self,
        weights: np.ndarray,
        *,
        max_weight: float,
    ) -> np.ndarray:

        weights = weights.copy()

        if max_weight >= 1:
            return weights

        while True:
            over = weights > max_weight

            if not over.any():
                break

            excess = float((weights[over] - max_weight).sum())

            weights[over] = max_weight

            under = ~over

            if not under.any():
                break

            under_total = weights[under].sum()

            if under_total <= 0:
                raise ValueError("Cannot redistribute excess weight.")

            weights[under] += excess * weights[under] / under_total

        return weights
