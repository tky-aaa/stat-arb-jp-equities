import numpy as np


class LeverageCalculator:
    """
    Calculate and constrain portfolio leverage.
    """

    def gross_leverage(
        self,
        weights: list[float],
    ) -> float:
        """
        Compute gross leverage.

        Gross leverage is:

            sum(|w_i|)
        """

        weights_array = np.asarray(
            weights,
            dtype=float,
        )

        return float(np.abs(weights_array).sum())

    def net_exposure(
        self,
        weights: list[float],
    ) -> float:
        """
        Compute net portfolio exposure.

        Net exposure is:

            sum(w_i)
        """

        weights_array = np.asarray(
            weights,
            dtype=float,
        )

        return float(weights_array.sum())

    def scale_to_leverage(
        self,
        weights: list[float],
        target_leverage: float,
    ) -> list[float]:
        """
        Scale portfolio weights to target gross leverage.
        """

        if target_leverage <= 0:
            raise ValueError("target_leverage must be positive.")

        weights_array = np.asarray(
            weights,
            dtype=float,
        )

        current_leverage = np.abs(weights_array).sum()

        if current_leverage == 0:
            raise ValueError("Cannot scale zero portfolio.")

        scaled = weights_array * target_leverage / current_leverage

        return scaled.tolist()
