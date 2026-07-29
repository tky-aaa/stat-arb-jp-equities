from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CandidateGroup:
    """
    Group of instruments passed to cointegration analysis.
    """

    tickers: list[str]


def generate_candidate_groups(
    clusters: pd.Series,
    min_size: int = 2,
    max_size: int = 20,
) -> list[CandidateGroup]:
    """
    Generate candidate groups from clustering result.

    Parameters
    ----------
    clusters:
        Cluster labels.

        index:
            ticker

        values:
            cluster id

    min_size:
        Minimum number of instruments.

    max_size:
        Maximum number of instruments.

    Returns
    -------
    list[CandidateGroup]
    """

    groups = []

    for cluster_id in clusters.unique():
        tickers = clusters[clusters == cluster_id].index.tolist()

        size = len(tickers)

        if min_size <= size <= max_size:
            groups.append(CandidateGroup(tickers=tickers))

    return groups


def extract_group_prices(
    prices: pd.DataFrame,
    group: CandidateGroup,
) -> pd.DataFrame:
    """
    Extract price series for a candidate group.

    Parameters
    ----------
    prices:
        Price DataFrame.

        columns:
            ticker

    group:
        CandidateGroup

    Returns
    -------
    pd.DataFrame
    """

    return prices[group.tickers]
