import numpy as np


def gross_leverage(
    weights: list[float],
) -> float:
    """
    Compute gross leverage.

    sum(|w_i|)
    """

    return float(np.abs(np.array(weights)).sum())


def net_exposure(
    weights: list[float],
) -> float:
    """
    Compute net market exposure.

    sum(w_i)
    """

    return float(np.sum(np.array(weights)))


def scale_to_leverage(
    weights: list[float],
    *,
    target_leverage: float = 1.0,
) -> list[float]:
    """
    Scale portfolio to target gross leverage.
    """

    if target_leverage <= 0:
        raise ValueError("target_leverage must be positive.")

    weights_array = np.array(
        weights,
        dtype=float,
    )

    current = np.abs(weights_array).sum()

    if current == 0:
        raise ValueError("Cannot scale zero portfolio.")

    return (weights_array * target_leverage / current).tolist()
