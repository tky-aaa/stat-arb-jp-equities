import logging

import pandas as pd

from statarb.cointegration.johansen import (
    JohansenResult,
    estimate_cointegration,
)
from statarb.screening.candidate import CandidateGroup
from statarb.screening.subgroups import enumerate_subgroups

logger = logging.getLogger(__name__)


def search_cointegrated_subgroups(
    group: CandidateGroup,
    log_prices: pd.DataFrame,
    *,
    min_assets: int = 2,
    max_assets: int = 5,
) -> list[JohansenResult]:
    """
    Search all cointegrated subgroups.
    """

    results: list[JohansenResult] = []

    subgroups = enumerate_subgroups(
        group,
        min_assets=min_assets,
        max_assets=max_assets,
    )

    for subgroup in subgroups:
        prices = log_prices[subgroup.tickers]

        try:
            result = estimate_cointegration(
                prices,
            )

        except ValueError as error:
            logger.debug(
                "Johansen failed for %s: %s",
                subgroup.tickers,
                error,
            )
            continue

        if result.rank > 0:
            results.append(result)

    return results
