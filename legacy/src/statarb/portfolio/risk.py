import numpy as np


def cap_weights(
    weights: list[float],
    *,
    max_weight: float = 0.2,
) -> list[float]:
    """
    Limit maximum allocation per asset.

    Keeps total weight normalized.
    """

    if max_weight <= 0:
        raise ValueError("max_weight must be positive.")

    weights_array = np.array(
        weights,
        dtype=float,
    )

    if weights_array.sum() <= 0:
        raise ValueError("Weights must have positive sum.")

    weights_array = weights_array / weights_array.sum()

    while True:
        over = weights_array > max_weight

        if not over.any():
            break

        excess = (weights_array[over] - max_weight).sum()

        weights_array[over] = max_weight

        under = ~over

        if not under.any():
            break

        weights_array[under] += (
            excess * weights_array[under] / weights_array[under].sum()
        )

    return weights_array.tolist()


def apply_leverage_limit(
    weights: list[float],
    *,
    max_leverage: float = 1.0,
) -> list[float]:
    """
    Normalize weights to satisfy gross leverage constraint.

    Gross leverage:
        sum(|w_i|)
    """

    if max_leverage <= 0:
        raise ValueError("max_leverage must be positive.")

    weights_array = np.array(weights)

    leverage = np.abs(weights_array).sum()

    if leverage <= max_leverage:
        return weights_array.tolist()

    return (weights_array * max_leverage / leverage).tolist()
