import numpy as np

from statarb.config.contract import CointegrationAnalysis


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

        print("[PortfolioAllocator] before cap:", weights)
        print("[PortfolioAllocator] max_weight:", max_weight)

        weights = self._cap_weights(
            weights,
            max_weight=max_weight,
        )

        print("[PortfolioAllocator] after cap:", weights)

        return weights.tolist()

    def _cap_weights(
        self,
        weights: np.ndarray,
        *,
        max_weight: float,
    ) -> np.ndarray:

        weights = np.asarray(weights, dtype=float).copy()

        if max_weight >= 1:
            return weights

        if max_weight * len(weights) < 1.0 - 1e-12:
            raise ValueError("max_weight is too small to allocate the full portfolio.")

        # Iteratively fix weights that exceed the cap.
        # Each iteration permanently fixes at least one weight,
        # so the algorithm terminates in at most len(weights) iterations.
        fixed = np.zeros(len(weights), dtype=bool)
        result = np.zeros(len(weights), dtype=float)

        remaining = 1.0

        while True:
            active = ~fixed

            if not active.any():
                break

            active_weights = weights[active]
            total = active_weights.sum()

            if total <= 0:
                raise ValueError("Cannot redistribute excess weight.")

            allocation = active_weights / total * remaining

            if np.all(allocation <= max_weight + 1e-12):
                result[active] = allocation
                break

            active_indices = np.flatnonzero(active)
            over = allocation > max_weight

            result[active_indices[over]] = max_weight
            fixed[active_indices[over]] = True

            remaining -= float(max_weight * over.sum())

        # Remove floating-point residue while preserving the cap.
        result = np.minimum(result, max_weight)

        residual = 1.0 - result.sum()

        if abs(residual) > 1e-12:
            available = result < max_weight - 1e-12

            if not available.any():
                raise ValueError("Cannot normalize capped weights.")

            available_indices = np.flatnonzero(available)

            for index in available_indices:
                capacity = max_weight - result[index]
                addition = min(capacity, residual)
                result[index] += addition
                residual -= addition

                if residual <= 1e-12:
                    break

        return result
