from itertools import combinations

from statarb.screening.candidate import CandidateGroup


def enumerate_subgroups(
    group: CandidateGroup,
    min_assets: int = 2,
    max_assets: int = 5,
) -> list[CandidateGroup]:
    """
    Enumerate all subgroups.

    Parameters
    ----------
    group:
        Original candidate group.

    min_assets:
        Minimum number of assets.

    max_assets:
        Maximum number of assets.

    Returns
    -------
    list[CandidateGroup]
    """

    tickers = group.tickers

    subgroups = []

    upper = min(
        max_assets,
        len(tickers),
    )

    for size in range(
        min_assets,
        upper + 1,
    ):
        for subset in combinations(
            tickers,
            size,
        ):
            subgroups.append(
                CandidateGroup(
                    tickers=list(subset),
                )
            )

    return subgroups
