import pandas as pd

from .evaluation import evaluate_spread


def rolling_spread_persistence(
    spread: pd.Series,
    *,
    window: int = 120,
) -> float:
    """
    Estimate rolling spread persistence.

    Fraction of rolling windows
    where the spread satisfies
    stationarity conditions.

    Parameters
    ----------
    spread:
        Spread time series.

    window:
        Rolling window length.

    Returns
    -------
    float
        Fraction of valid stationary windows.
    """

    spread = spread.dropna()

    if len(spread) < window:
        raise ValueError("Too few observations for rolling persistence.")

    total = 0
    success = 0

    for start in range(
        len(spread) - window + 1,
    ):
        sample = spread.iloc[start : start + window]

        evaluation = evaluate_spread(
            sample,
        )

        total += 1

        if evaluation.adf_pvalue < 0.05 and evaluation.kpss_pvalue > 0.05:
            success += 1

    return success / total
