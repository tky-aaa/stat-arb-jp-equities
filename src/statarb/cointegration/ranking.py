import pandas as pd


def _calculate_score(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate ranking score.

    Lower score is better.
    """

    df = df.copy()

    df["half_life_rank"] = (
        df["half_life"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    df["persistence_rank"] = (
        df["persistence"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    df["score"] = df["half_life_rank"] + df["persistence_rank"]

    return df


def rank_spreads(
    evaluation_results: pd.DataFrame,
    *,
    adf_threshold: float = 0.05,
    kpss_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Rank cointegration spreads.

    Each row represents one
    cointegration vector.
    """

    df = evaluation_results.copy()

    # =========================
    # Stationarity filter
    # =========================

    df = df[
        (df["adf_pvalue"] < adf_threshold) & (df["kpss_pvalue"] > kpss_threshold)
    ].copy()

    if len(df) == 0:
        return df

    # =========================
    # Remove invalid values
    # =========================

    df = df[
        df["half_life"].notna()
        & df["persistence"].notna()
        & (df["half_life"] != float("inf"))
    ].copy()

    if len(df) == 0:
        return df

    # =========================
    # Score
    # =========================

    df = _calculate_score(
        df,
    )

    # =========================
    # Sort
    # =========================

    df = df.sort_values(
        by=[
            "score",
            "half_life",
            "persistence",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    return df.reset_index(
        drop=True,
    )
